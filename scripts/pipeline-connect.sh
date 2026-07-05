#!/usr/bin/env bash
# One-shot connect to the remote TentNetFA pipeline UI, with the wireguard
# VPN scoped to THIS process only (github/slack/... stay on your normal
# connection):
#
#   1. builds a private network namespace and moves a wireguard interface
#      into it (the official wireguard netns pattern: the encrypted UDP
#      socket stays in your main namespace, so only processes started
#      inside the namespace route through the VPN);
#   2. starts the pipeline UI on the server (tmux, else screen, else
#      setsid+nohup) and opens an SSH tunnel from inside the namespace;
#   3. the tunnel lands on a unix socket (unix sockets are filesystem
#      objects, not network-namespaced) and a tiny python relay exposes it
#      as http://localhost:8501 for your normal browser.
#
# Ctrl+C tears down the tunnel, relay and namespace; the UI and any running
# pipelines keep running on the server.
#
# Assumes key-based ssh (agent already holding the key) and the host key
# already in known_hosts — the tunnel runs in the background and cannot
# answer interactive prompts.
#
# Configure via scripts/pipeline-connect-setup.sh (writes the gitignored .env).
set -euo pipefail

# Machine-specific settings come from, in order of precedence: the
# environment, the repo .env, then the defaults below. Only the whitelisted
# keys are read from .env - the rest of it (DATA_DIR, ...) is left alone.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$repo_root/.env"
if [[ -f "$env_file" ]]; then
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        [[ "$_line" =~ ^[[:space:]]*(#|$) ]] && continue
        _key="${_line%%=*}"; _val="${_line#*=}"
        [[ "$_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        _val="${_val%\"}"; _val="${_val#\"}"
        case "$_key" in
            WG_CONF | SSH_HOST | REMOTE_DIR | PORT | SESSION_NAME | NS | WG_IF)
                [[ -z "${!_key:-}" ]] && printf -v "$_key" '%s' "$_val" ;;
        esac
    done < "$env_file"
fi

WG_CONF="${WG_CONF:-}"                  # wg-quick name (/etc/wireguard/<name>.conf) or path to a .conf
SSH_HOST="${SSH_HOST:-}"                # user@server (address reachable through the VPN)
REMOTE_DIR="${REMOTE_DIR:-TentNetFA}"   # repo on the server: absolute, or relative to the remote home
PORT="${PORT:-8501}"
SESSION_NAME="${SESSION_NAME:-pipeline-ui}"
NS="${NS:-tentnet}"                     # network namespace name
WG_IF="${WG_IF:-wg-tnfa}"               # interface name (<= 15 chars)
SOCK="/tmp/tentnet-ui-${PORT}.sock"     # must stay short: AF_UNIX paths cap at ~108 chars

if [[ -z "$WG_CONF" || -z "$SSH_HOST" ]]; then
    echo "WG_CONF / SSH_HOST are not configured." >&2
    echo "Run scripts/pipeline-connect-setup.sh once (writes them to the gitignored .env)," >&2
    echo "or set them in the environment." >&2
    exit 1
fi

for tool in wg wg-quick ip python3 curl ssh; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done

if [[ -f "$WG_CONF" ]]; then conf_path="$WG_CONF"; else conf_path="/etc/wireguard/${WG_CONF}.conf"; fi

echo "sudo is needed for wireguard + network namespace setup:"
sudo -v
sudo test -r "$conf_path" || { echo "Cannot read wireguard config $conf_path" >&2; exit 1; }

if curl -sf -o /dev/null --max-time 2 "http://localhost:$PORT"; then
    echo "Something already listens on localhost:$PORT - is a previous session still up?" >&2
    exit 1
fi

# poetry's dir is missing from non-interactive remote PATHs, so the manual
# stop command has to carry the same export the launch/stop helpers use
manual_stop_hint() {
    echo "ssh $SSH_HOST 'export PATH=\"\$HOME/.local/bin:\$HOME/.poetry/bin:\$PATH\"; cd $REMOTE_DIR && poetry run pipeline-ui-stop'"
}

# ----- teardown (also clears leftovers from a crashed earlier run) -----------
relay_pid=""
backend_started=""
stop_backend() {
    echo "Stopping pipeline UI backend on $SSH_HOST..."
    in_ns ssh "$SSH_HOST" "RDIR='$REMOTE_DIR' bash -s" <<'STOPREMOTE' \
        || echo "Backend stop failed - stop it manually: $(manual_stop_hint)" >&2
export PATH="$HOME/.local/bin:$HOME/.poetry/bin:$PATH"
cd "$RDIR" && poetry run pipeline-ui-stop
STOPREMOTE
}
cleanup() {
    # offer backend teardown while the namespace (and thus the VPN) still exists
    if [[ -n "$backend_started" && -r /dev/tty && -w /dev/tty ]]; then
        backend_started=""
        local answer=""
        read -r -t 60 -p "Also stop the pipeline UI on $SSH_HOST? Running pipeline stages die with it. [y/N] " \
            answer </dev/tty || true
        [[ "$answer" =~ ^[Yy]$ ]] && stop_backend
    fi
    [[ -n "$relay_pid" ]] && kill "$relay_pid" 2>/dev/null || true
    if sudo ip netns list 2>/dev/null | grep -qw "$NS"; then
        # kill the tunnel ssh (and anything else) inside the namespace first:
        # interfaces in a namespace are only destroyed once its processes exit
        sudo ip netns pids "$NS" 2>/dev/null | xargs -r sudo kill 2>/dev/null || true
        sleep 1
        sudo ip netns del "$NS" 2>/dev/null || true
    fi
    sudo rm -rf "/etc/netns/$NS" 2>/dev/null || true
    sudo ip link del "$WG_IF" 2>/dev/null || true  # only exists here if setup died mid-way
    rm -f "$SOCK"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
cleanup   # stale state from a crashed run

# ----- namespace + wireguard --------------------------------------------------
echo "Building network namespace '$NS' with wireguard '$WG_IF'..."
conf_text="$(sudo cat "$conf_path")"
addresses="$(grep -im1 '^[[:space:]]*Address' <<<"$conf_text" | cut -d= -f2- | tr ',' ' ' || true)"
dns_servers="$(grep -im1 '^[[:space:]]*DNS' <<<"$conf_text" | cut -d= -f2- | tr ',' ' ' || true)"
mtu="$(grep -im1 '^[[:space:]]*MTU' <<<"$conf_text" | cut -d= -f2- | tr -d '[:space:]' || true)"
[[ -n "$addresses" ]] || { echo "No Address= line found in $conf_path" >&2; exit 1; }

sudo ip netns add "$NS"
sudo ip -n "$NS" link set lo up
# created in the MAIN namespace so the encrypted UDP socket stays out here...
sudo ip link add "$WG_IF" type wireguard
# single root shell: wg re-opens its config via /proc, which fails if the
# pipe belongs to the unprivileged user (root DAC override is restricted here)
sudo bash -c "wg setconf $(printf %q "$WG_IF") <(wg-quick strip $(printf %q "$conf_path"))"
# ...then only the cleartext side moves into the namespace
sudo ip link set "$WG_IF" netns "$NS"
for addr in $addresses; do
    sudo ip -n "$NS" addr add "$addr" dev "$WG_IF"
done
[[ -n "$mtu" ]] && sudo ip -n "$NS" link set "$WG_IF" mtu "$mtu"
sudo ip -n "$NS" link set "$WG_IF" up
sudo ip -n "$NS" route add default dev "$WG_IF"
if [[ "$addresses" == *:* ]]; then
    sudo ip -n "$NS" -6 route add default dev "$WG_IF" || true
fi

# DNS inside the namespace: the wg config's DNS= if present, else the real
# upstream resolvers (the systemd-resolved stub on 127.0.0.53 is unreachable
# from inside the namespace). Irrelevant if SSH_HOST is an IP.
sudo mkdir -p "/etc/netns/$NS"
if [[ -n "$dns_servers" ]]; then
    printf 'nameserver %s\n' $dns_servers | sudo tee "/etc/netns/$NS/resolv.conf" >/dev/null
elif [[ -f /run/systemd/resolve/resolv.conf ]]; then
    sudo cp /run/systemd/resolve/resolv.conf "/etc/netns/$NS/resolv.conf"
fi

# run a command inside the namespace, as you, with the ssh agent available
in_ns() {
    sudo --preserve-env=SSH_AUTH_SOCK ip netns exec "$NS" \
        sudo --preserve-env=SSH_AUTH_SOCK -u "$USER" -- "$@"
}

# ----- server: start the pipeline UI (tmux -> screen -> nohup) ----------------
echo "Starting pipeline UI on $SSH_HOST..."
in_ns ssh "$SSH_HOST" "SESSION='$SESSION_NAME' RDIR='$REMOTE_DIR' RPORT='$PORT' bash -s" <<'REMOTE'
# non-interactive shells often miss poetry's install dir (.bashrc exits
# early before its PATH line), so prepend the usual locations explicitly
launch="export PATH=\"\$HOME/.local/bin:\$HOME/.poetry/bin:\$PATH\"; cd \"$RDIR\" && poetry run pipeline-ui --remote --server.port $RPORT"
if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  reusing tmux session '$SESSION'."
    else
        tmux new-session -d -s "$SESSION" "bash -lc '$launch'"
        echo "  UI launched in tmux session '$SESSION'."
    fi
elif command -v screen >/dev/null 2>&1; then
    if screen -ls 2>/dev/null | grep -q "[.]$SESSION[[:space:]]"; then
        echo "  reusing screen session '$SESSION'."
    else
        screen -dmS "$SESSION" bash -lc "$launch"
        echo "  UI launched in screen session '$SESSION' (no tmux found)."
    fi
else
    if pgrep -f "streamlit run .*displacement_tracker/pipelines/app.py" >/dev/null 2>&1; then
        echo "  reusing already-running UI (no tmux/screen found)."
    else
        log="$RDIR/pipeline-ui.nohup.log"
        setsid nohup bash -lc "$launch" >>"$log" 2>&1 </dev/null &
        echo "  UI launched detached via setsid+nohup (no tmux/screen found); log: $log"
    fi
fi
REMOTE
backend_started=1

# ----- tunnel: ssh (inside ns) -> unix socket -> python relay -> localhost -----
echo "Opening tunnel localhost:$PORT -> $SSH_HOST:$PORT..."
rm -f "$SOCK"
in_ns ssh -N -o ExitOnForwardFailure=yes -o StreamLocalBindUnlink=yes \
    -L "$SOCK:localhost:$PORT" "$SSH_HOST" &
tunnel_pid=$!

for _ in $(seq 1 30); do
    [[ -S "$SOCK" ]] && break
    kill -0 "$tunnel_pid" 2>/dev/null || { echo "Tunnel ssh died." >&2; exit 1; }
    sleep 1
done
[[ -S "$SOCK" ]] || { echo "Tunnel socket never appeared." >&2; exit 1; }

python3 - "$PORT" "$SOCK" <<'PY' &
import asyncio, sys
port, sock = int(sys.argv[1]), sys.argv[2]

async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

async def handle(client_r, client_w):
    try:
        up_r, up_w = await asyncio.open_unix_connection(sock)
    except Exception:
        client_w.close()
        return
    await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))

async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", port)
    async with server:
        await server.serve_forever()

asyncio.run(main())
PY
relay_pid=$!

echo -n "Waiting for the UI to answer"
for _ in $(seq 1 60); do
    if curl -sf -o /dev/null "http://localhost:$PORT"; then
        echo " - up."
        break
    fi
    kill -0 "$tunnel_pid" 2>/dev/null || { echo; echo "Tunnel died." >&2; exit 1; }
    echo -n "."
    sleep 1
done

xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || echo "Open http://localhost:$PORT in your browser."

echo
echo "Connected. Only this tunnel uses the VPN; everything else is untouched."
echo "Ctrl+C to disconnect (UI keeps running on the server)."
echo "To stop the UI later (needs the VPN, so easiest while this script runs): $(manual_stop_hint)"
wait "$tunnel_pid" || true
