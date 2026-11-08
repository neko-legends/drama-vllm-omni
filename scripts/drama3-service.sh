#!/usr/bin/env bash
# drama3-service.sh — start|stop|status|restart the Drama 3 NVFP4 engine on ripper.
#
# Safe for a UI toggle: every verb is idempotent, `start` refuses to touch a port
# that someone else already answers on, `stop` only stops the transient user unit
# this script started (verified by the unit's MainPID and its command line), and
# the engine runs under a GPU flock so it cannot share the card with another job.
#
#   ./drama3-service.sh start     # idempotent; refuses if :18793 is already answering
#   ./drama3-service.sh status    # exit 0 = answering, 1 = not answering, 2 = not ours
#   ./drama3-service.sh restart
#   ./drama3-service.sh stop      # idempotent; refuses if the server is not ours
#
# Locks: /tmp/drama3-gpu.lock (created here) and, when present, the host's shared
# GPU lock /tmp/gpu.lock. The unit holds both for the server's whole lifetime, so
# a second GPU job that takes either file is refused instead of colliding.
#
# Environment overrides: DRAMA3_ROOT, DRAMA3_BIN, DRAMA3_CHECKPOINT, DRAMA3_RUNTIME,
# DRAMA3_HOST, DRAMA3_PORT, DRAMA3_UNIT, DRAMA3_GPU_LOCK, DRAMA3_HOST_GPU_LOCK,
# DRAMA3_MEMORY_MAX_BYTES, DRAMA3_READY_TIMEOUT, DRAMA3_GPU.
set -uo pipefail

ROOT=${DRAMA3_ROOT:-/home/jun/drama3}
SRC=$ROOT/recipe-src
BIN=${DRAMA3_BIN:-$SRC/build/tensorfold-nvfp4/zig-out/native/bin/tensorfold-native}
CKPT=${DRAMA3_CHECKPOINT:-$SRC/drama-3-NVFP4-Pruned}
RUNTIME=${DRAMA3_RUNTIME:-$SRC/runtime}
HOST=${DRAMA3_HOST:-127.0.0.1}
PORT=${DRAMA3_PORT:-18793}
UNIT=${DRAMA3_UNIT:-drama3-service.service}
LOCK=${DRAMA3_GPU_LOCK:-/tmp/drama3-gpu.lock}
HOST_LOCK=${DRAMA3_HOST_GPU_LOCK:-/tmp/gpu.lock}
MEM_MAX=${DRAMA3_MEMORY_MAX_BYTES:-57982058496}
READY_TIMEOUT=${DRAMA3_READY_TIMEOUT:-180}
GPU=${DRAMA3_GPU:-0}
STATE=$ROOT/.drama3-service.state
LOGDIR=$ROOT/logs
LOG=$LOGDIR/service.log
MODEL=fishaudio/drama-3
SELF=$(readlink -f "$0" 2>/dev/null || echo "$0")
SERVE_ARGS=(serve "$CKPT" --host "$HOST" --port "$PORT" --name "$MODEL"
            --context 8192 --parallel 1 --no-drafts --prompt-cache-gib 0
            --keep-warm 0 --cfg-gamma 1.5 --cfg-batched)
URL=http://$HOST:$PORT

say() { printf 'drama3-service: %s\n' "$*"; }
die() { say "$1" >&2; exit "${2:-1}"; }

health() { curl -fsS -m 2 -o /dev/null "$URL/health" 2>/dev/null; }
health_ms() { curl -s -m 2 -o /dev/null -w '%{time_total}' "$URL/health" 2>/dev/null; }
unit_active() { [[ $(systemctl --user is-active "$UNIT" 2>/dev/null) == active ]]; }
unit_pid() { systemctl --user show -p MainPID --value "$UNIT" 2>/dev/null; }
unit_state() { local s; s=$(systemctl --user is-active "$UNIT" 2>/dev/null | head -1); printf '%s' "${s:-inactive}"; }
cmdline() { [[ -r /proc/$1/cmdline ]] && tr '\0' ' ' < "/proc/$1/cmdline"; }

# ours = the process on $PORT is the binary this script starts, serving this port
is_ours() {
    local pid=${1:-} line
    [[ -n $pid && $pid != 0 && -r /proc/$pid/cmdline ]] || return 1
    line=$(cmdline "$pid")
    [[ $line == *"$BIN"* && $line == *" --port $PORT "* ]]
}

# pid of whatever listens on $PORT, if it can be seen
port_pid() {
    fuser -n tcp "$PORT" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | head -1
}

lock_free() {  # $1 = lock file; true when nobody holds it
    [[ -e $1 ]] || return 0
    flock -n "$1" -c true 2>/dev/null
}

lock_holder() { fuser "$1" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | head -1; }

vram_of() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F', ' -v p="$1" '$1 == p {print $2}'
}

# The engine itself: runs under systemd as this script, takes both GPU locks,
# then execs the server so the locks live exactly as long as the server does.
__exec() {
    mkdir -p "$LOGDIR"
    touch "$LOCK"
    exec 9>>"$LOCK" || die "cannot open $LOCK" 73
    if ! flock -n 9; then
        say "GPU lock $LOCK is held by pid $(lock_holder "$LOCK") — refusing to start" >&2
        exit 75
    fi
    if [[ -e $HOST_LOCK ]]; then
        exec 8>>"$HOST_LOCK" || die "cannot open $HOST_LOCK" 73
        if ! flock -n 8; then
            say "host GPU lock $HOST_LOCK is held by pid $(lock_holder "$HOST_LOCK") — refusing to start" >&2
            exit 76
        fi
    fi
    cd "$RUNTIME" || die "no runtime directory $RUNTIME" 73
    exec env -i \
        HOME="$HOME" \
        PATH="/usr/local/cuda/bin:/usr/bin:/bin" \
        LANG=C.UTF-8 \
        PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES="$GPU" \
        HF_HOME="$RUNTIME/hf-cache" \
        HF_HUB_OFFLINE=1 \
        XDG_CACHE_HOME="$RUNTIME/runtime-cache" \
        TRITON_CACHE_DIR="$RUNTIME/runtime-cache/triton" \
        TORCHINDUCTOR_CACHE_DIR="$RUNTIME/runtime-cache/inductor" \
        CUDA_CACHE_PATH="$RUNTIME/runtime-cache/cuda" \
        FLASHINFER_WORKSPACE_BASE="$RUNTIME/runtime-cache/flashinfer" \
        TENSORFOLD_MEMORY_RESERVE_GIB=8 \
        TENSORFOLD_CUDA_MEMORY_LIMIT_GB=54 \
        "$BIN" "${SERVE_ARGS[@]}"
}

wait_ready() {
    local started=$SECONDS pid
    while (( SECONDS - started < READY_TIMEOUT )); do
        if health; then
            pid=$(unit_pid)
            printf '%s %s %s\n' "unit=$UNIT" "pid=$pid" "started_epoch=$(date +%s)" > "$STATE"
            printf 'bin=%s\nport=%s\n' "$BIN" "$PORT" >> "$STATE"
            say "ready on $URL after $((SECONDS - started))s (unit $UNIT, pid $pid)"
            return 0
        fi
        if ! unit_active; then
            say "unit $UNIT exited before answering; last log lines:" >&2
            tail -n 12 "$LOG" >&2 2>/dev/null
            journalctl --user -u "$UNIT" -n 12 --no-pager -o cat >&2 2>/dev/null
            systemctl --user reset-failed "$UNIT" >/dev/null 2>&1
            return 1
        fi
        sleep 0.3
    done
    say "not ready after ${READY_TIMEOUT}s; unit $UNIT is still active" >&2
    return 1
}

start() {
    local pid f
    [[ -x $BIN ]] || die "engine binary not found: $BIN" 4
    [[ -d $CKPT ]] || die "checkpoint directory not found: $CKPT" 4

    if unit_active; then
        pid=$(unit_pid)
        if is_ours "$pid"; then
            if health; then
                say "already running (unit $UNIT, pid $pid) — nothing to do"
                return 0
            fi
            say "unit $UNIT (pid $pid) is active but not answering yet; waiting"
            wait_ready && return 0 || return 1
        fi
        die "unit $UNIT is active with pid $pid, which is not this engine — refusing to start" 4
    fi

    if health; then
        pid=$(port_pid)
        say "refusing to start: $URL already answers (pid ${pid:-unknown}: $(cmdline "${pid:-0}" | cut -c1-120))" >&2
        return 2
    fi

    mkdir -p "$LOGDIR"
    touch "$LOCK"
    if ! lock_free "$LOCK"; then
        die "GPU lock $LOCK is held by pid $(lock_holder "$LOCK") — refusing to start" 3
    fi
    if [[ -e $HOST_LOCK ]] && ! lock_free "$HOST_LOCK"; then
        die "host GPU lock $HOST_LOCK is held by pid $(lock_holder "$HOST_LOCK") — refusing to start" 3
    fi

    systemd-run --user --unit="$UNIT" --collect \
        --property=MemoryMax="$MEM_MAX" \
        --property=MemorySwapMax=0 \
        --property=KillMode=control-group \
        --property=OOMPolicy=kill \
        --property=TimeoutStopSec=20 \
        --property=StandardOutput=append:"$LOG" \
        --property=StandardError=append:"$LOG" \
        "$SELF" __exec >/dev/null || die "systemd-run failed; is this a user systemd session?" 1

    say "started unit $UNIT; log $LOG"
    wait_ready
}

stop() {
    local pid started=$SECONDS
    if ! unit_active; then
        if health; then
            pid=$(port_pid)
            say "refusing to stop: $URL is answered by pid ${pid:-unknown}, which this script did not start" >&2
            return 2
        fi
        rm -f "$STATE"
        say "not running — nothing to do"
        return 0
    fi
    pid=$(unit_pid)
    if ! is_ours "$pid"; then
        say "refusing to stop: unit $UNIT runs pid $pid, not $BIN on port $PORT" >&2
        return 4
    fi
    systemctl --user stop "$UNIT" >/dev/null 2>&1
    while (( SECONDS - started < 40 )); do
        unit_active || { health || break; }
        sleep 0.25
    done
    if unit_active; then
        die "unit $UNIT is still active after 40s" 1
    fi
    if health; then
        die "$URL still answers after stopping $UNIT" 1
    fi
    rm -f "$STATE"
    say "stopped unit $UNIT (pid $pid); $URL no longer answers"
    return 0
}

status() {
    local pid state=stopped
    if unit_active; then
        pid=$(unit_pid)
        if is_ours "$pid"; then state=running; else state=foreign-unit; fi
    elif health; then
        state=foreign-server
        pid=$(port_pid)
    fi
    printf 'state:          %s\n' "$state"
    printf 'unit:           %s (%s)\n' "$UNIT" "$(unit_state)"
    printf 'port:           %s %s\n' "$URL" "$(health && echo answering || echo closed)"
    [[ -n ${pid:-} ]] && printf 'pid:            %s\n' "$pid"
    [[ -n ${pid:-} ]] && printf 'vram:           %s\n' "$(vram_of "$pid" || echo unknown)"
    for f in "$LOCK" "$HOST_LOCK"; do
        [[ -e $f ]] || continue
        if lock_free "$f"; then
            printf 'lock:           %s free\n' "$f"
        else
            printf 'lock:           %s held by pid %s\n' "$f" "$(lock_holder "$f")"
        fi
    done
    if [[ -r /proc/${pid:-0}/cmdline ]]; then
        printf 'cmdline:        %s\n' "$(cmdline "$pid")"
    fi
    if [[ $state == foreign-server ]]; then
        say "note: a server this script did not start owns the port; start/stop will refuse" >&2
    fi
    [[ $state == running || $state == foreign-server ]] && return 0
    return 1
}

usage() { echo "usage: ${0##*/} {start|stop|restart|status}" >&2; exit 64; }

case ${1:-} in
    start) start ;;
    stop) stop ;;
    restart) stop || exit $?; start ;;
    status) status ;;
    __exec) __exec ;;
    *) usage ;;
esac
