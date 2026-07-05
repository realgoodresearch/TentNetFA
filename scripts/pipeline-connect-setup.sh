#!/usr/bin/env bash
# Interactive one-time setup for pipeline-connect.sh: prompts for the
# machine-specific values and writes them to the repo .env (gitignored).
# Re-running offers the current values as defaults; unrelated .env keys
# (DATA_DIR, ...) are left untouched.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$repo_root/.env"

current() {  # last assignment in .env wins; optional surrounding quotes stripped
    local val
    [[ -f "$env_file" ]] || return 0
    val="$(grep -E "^$1=" "$env_file" | tail -n1 | cut -d= -f2-)" || return 0
    val="${val%\"}"; val="${val#\"}"
    printf '%s' "$val"
}

set_kv() {
    local key="$1" val="$2"
    touch "$env_file"
    if grep -qE "^${key}=" "$env_file"; then
        sed -i "s|^${key}=.*|${key}=\"${val}\"|" "$env_file"
    else
        printf '%s="%s"\n' "$key" "$val" >> "$env_file"
    fi
}

ask() {
    local key="$1" desc="$2" fallback="$3" required="${4:-}" def answer
    def="$(current "$key")"
    def="${def:-$fallback}"
    while true; do
        if [[ -n "$def" ]]; then
            read -r -p "$desc [$def]: " answer
            answer="${answer:-$def}"
        else
            read -r -p "$desc: " answer
        fi
        if [[ -n "$answer" || -z "$required" ]]; then
            break
        fi
        echo "  A value is required."
    done
    set_kv "$key" "$answer"
}

echo "Configuring pipeline-connect.sh - values are written to $env_file"
echo "(gitignored; press Enter to accept the [default])."
echo
ask WG_CONF      "Wireguard config: wg-quick name in /etc/wireguard, or path to a .conf" "" required
ask SSH_HOST     "SSH host (user@server or an ssh-config alias, reachable through the VPN)" "" required
ask REMOTE_DIR   "Repo directory on the server (absolute, or relative to the remote home)" "TentNetFA"
ask PORT         "UI port" "8501"
ask SESSION_NAME "Remote tmux/screen session name" "pipeline-ui"
ask NS           "Local network namespace name" "tentnet"
ask WG_IF        "Local wireguard interface name (max 15 chars)" "wg-tnfa"
echo
echo "Done. Connect with:  scripts/pipeline-connect.sh"
