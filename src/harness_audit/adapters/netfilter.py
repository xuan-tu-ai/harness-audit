"""Network gate for isolated runs (used by isolate.sh; standard library only).

``serve SOCK ALLOW UPSTREAM``: outside the run's network namespace, accept HTTP ``CONNECT host:port``
requests on the Unix socket SOCK and forward only those whose host is in ALLOW (comma-separated
exact names) and port is 443, directly or through the HTTP proxy UPSTREAM (``http://host:port``,
or ``-`` for none). Everything else gets ``403``.

``bridge PORT SOCK``: inside the namespace (no other network), bring ``lo`` up and relay TCP
connections on 127.0.0.1:PORT to SOCK, so the CLI can use ``HTTPS_PROXY=http://127.0.0.1:PORT``.

The agent's own commands inherit the same proxy, so they too can reach only the allowed hosts: the
model gateway, never a site that hosts the benchmark's tests or solutions.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import socket
import struct
import sys
from urllib.parse import urlsplit


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        with contextlib.suppress(Exception):
            w.close()


async def splice(r1: asyncio.StreamReader, w1: asyncio.StreamWriter, r2: asyncio.StreamReader, w2: asyncio.StreamWriter) -> None:
    await asyncio.gather(pipe(r1, w2), pipe(r2, w1))


async def read_head(r: asyncio.StreamReader) -> bytes:
    return await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), timeout=30)


def allowed(target: str, allow: set[str]) -> bool:
    host, _, port = target.rpartition(":")
    return port == "443" and host.lower() in allow


async def serve(sock: str, allow_csv: str, upstream: str) -> None:
    allow = {h.strip().lower() for h in allow_csv.split(",") if h.strip()}
    up = None if upstream in ("", "-") else urlsplit(upstream)

    async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            head = await read_head(r)
            parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
            if len(parts) < 2 or parts[0].upper() != "CONNECT" or not allowed(parts[1], allow):
                w.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                await w.drain()
                w.close()
                return
            target = parts[1]
            if up is None:
                host, _, port = target.rpartition(":")
                r2, w2 = await asyncio.open_connection(host, int(port))
            else:
                r2, w2 = await asyncio.open_connection(up.hostname, up.port or 80)
                w2.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
                await w2.drain()
                reply = await read_head(r2)
                if b" 200" not in reply.split(b"\r\n", 1)[0]:
                    w.write(reply)
                    await w.drain()
                    w.close()
                    w2.close()
                    return
            w.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await w.drain()
            await splice(r, w, r2, w2)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError, OSError):
            with contextlib.suppress(Exception):
                w.close()

    server = await asyncio.start_unix_server(handle, path=sock)
    async with server:
        await server.serve_forever()


def loopback_up() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # SIOCSIFFLAGS: IFF_UP | IFF_LOOPBACK | IFF_RUNNING
    fcntl.ioctl(s, 0x8914, struct.pack("16sH14x", b"lo", 0x1 | 0x8 | 0x40))
    s.close()


async def bridge(port: int, sock: str) -> None:
    async def handle(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            r2, w2 = await asyncio.open_unix_connection(sock)
        except OSError:
            w.close()
            return
        await splice(r, w, r2, w2)

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    async with server:
        await server.serve_forever()


def main(argv: list[str]) -> None:
    if argv[:1] == ["serve"] and len(argv) == 4:
        asyncio.run(serve(argv[1], argv[2], argv[3]))
    elif argv[:1] == ["bridge"] and len(argv) == 3:
        loopback_up()
        asyncio.run(bridge(int(argv[1]), argv[2]))
    else:
        raise SystemExit("usage: netfilter.py serve SOCK ALLOW UPSTREAM | bridge PORT SOCK")


if __name__ == "__main__":
    main(sys.argv[1:])
