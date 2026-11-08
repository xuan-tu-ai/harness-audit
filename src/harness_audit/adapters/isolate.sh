#!/bin/sh
# Run the Claude CLI so that the agent can see only its own working directory and reach only the
# model endpoint.
#
# Used as the Claude Agent SDK's cli_path when the adapter's `isolate` option is on (Linux, root, or a
# user allowed to create namespaces). The CLI starts in new mount, PID and network namespaces:
#
# - files: /home, /root, /tmp and /var/tmp are empty tmpfs mounts, except for the run's working
#   directory (the SDK's cwd) and CLAUDE_CONFIG_DIR, which are bind-mounted back at the same paths.
#   Hidden tests, the task repository, the run cache and other runs' working directories are not
#   reachable, not even through /proc/<pid>/root of processes outside (new PID namespace).
# - network: the namespace has only loopback. netfilter.py relays 127.0.0.1:18080 to a gate outside
#   that allows CONNECT to the hosts in HARNESS_AUDIT_ALLOW_HOSTS (port 443) and nothing else, through
#   the host's own HTTPS proxy if one is set. The CLI and every command the agent runs use that proxy.
#
# Environment: HARNESS_AUDIT_CLAUDE_BIN (the real CLI), HARNESS_AUDIT_ALLOW_HOSTS (comma-separated),
# HARNESS_AUDIT_KEEP_FILES (colon-separated files under hidden directories to copy back in, e.g. the
# CA bundle a TLS-inspecting proxy needs).
set -eu
WORK=$(pwd -P)
CFG=${CLAUDE_CONFIG_DIR:?CLAUDE_CONFIG_DIR must be set}
BIN=${HARNESS_AUDIT_CLAUDE_BIN:?HARNESS_AUDIT_CLAUDE_BIN must be set}
ALLOW=${HARNESS_AUDIT_ALLOW_HOSTS:?HARNESS_AUDIT_ALLOW_HOSTS must be set}
UP=${HTTPS_PROXY:-${https_proxy:--}}
HERE=$(dirname "$(readlink -f "$0")")
NETDIR=$(mktemp -d /run/harness-audit-net.XXXXXX)
cp "$HERE/netfilter.py" "$NETDIR/netfilter.py"
python3 -I "$NETDIR/netfilter.py" serve "$NETDIR/gate.sock" "$ALLOW" "$UP" &
GATE=$!
cleanup() { kill "$GATE" 2>/dev/null || true; rm -rf "$NETDIR"; }
trap cleanup EXIT
trap 'exit 143' INT TERM
i=0
while [ ! -S "$NETDIR/gate.sock" ]; do
  i=$((i + 1)); [ "$i" -gt 100 ] && { echo "isolate: network gate did not start" >&2; exit 1; }
  sleep 0.05
done
export WORK CFG BIN NETDIR
set +e
unshare --mount --pid --fork --mount-proc --net --propagation private /bin/sh -eu -c '
python3 -I "$NETDIR/netfilter.py" bridge 18080 "$NETDIR/gate.sock" &
i=0
until python3 -I -c "import socket; socket.create_connection((\"127.0.0.1\", 18080), 1).close()" 2>/dev/null; do
  i=$((i + 1)); [ "$i" -gt 100 ] && { echo "isolate: network bridge did not start" >&2; exit 1; }
  sleep 0.05
done
P=http://127.0.0.1:18080
export HTTPS_PROXY=$P https_proxy=$P HTTP_PROXY=$P http_proxy=$P
unset NO_PROXY no_proxy ALL_PROXY all_proxy
S="$NETDIR/stage"; mkdir "$S"
mount -t tmpfs none "$S"
mkdir "$S/work" "$S/cfg" "$S/keep"
mount --bind "$WORK" "$S/work"
mount --bind "$CFG" "$S/cfg"
IFS=:
for f in ${HARNESS_AUDIT_KEEP_FILES:-}; do
  [ -f "$f" ] && mkdir -p "$S/keep$(dirname "$f")" && cp "$f" "$S/keep$f"
done
unset IFS
for d in /home /root /tmp /var/tmp; do
  [ -d "$d" ] && mount -t tmpfs -o mode=1777 none "$d"
done
mkdir -p "$WORK" "$CFG"
mount --bind "$S/work" "$WORK"
mount --bind "$S/cfg" "$CFG"
( cd "$S/keep" && find . -type f ) | while read -r f; do
  mkdir -p "$(dirname "/$f")" && cp "$S/keep/$f" "/$f"
done
cd "$WORK"
exec "$BIN" "$@"
' sh "$@"
status=$?
exit "$status"
