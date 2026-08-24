#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.eagleeye.streamlit"
DOMAIN="gui/$(id -u)/${LABEL}"
LOG_DIR="${EAGLE_EYE_LOG_DIR:-$HOME/Library/Logs/EagleEye}"
STDOUT_LOG="$LOG_DIR/streamlit.log"
STDERR_LOG="$LOG_DIR/streamlit.error.log"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-127.0.0.1}"
ACTION="${1:-status}"

health_url="http://${ADDRESS}:${PORT}/_stcore/health"

start_service() {
  mkdir -p "$LOG_DIR"
  : > "$STDOUT_LOG"
  : > "$STDERR_LOG"
  launchctl remove "$LABEL" 2>/dev/null || true
  for _ in {1..20}; do
    if ! launchctl print "$DOMAIN" >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done

  local command
  printf -v command 'cd %q && exec env PYTHONFAULTHANDLER=1 PORT=%q ADDRESS=%q ./run_streamlit.sh' \
    "$ROOT_DIR" "$PORT" "$ADDRESS"
  launchctl submit -l "$LABEL" -o "$STDOUT_LOG" -e "$STDERR_LOG" -- /bin/zsh -lc "$command"

  for _ in {1..60}; do
    if curl --silent --fail "$health_url" >/dev/null 2>&1; then
      echo "Eagle Eye is running at http://${ADDRESS}:${PORT}"
      echo "Logs: $STDOUT_LOG and $STDERR_LOG"
      return 0
    fi
    sleep 0.25
  done

  echo "Eagle Eye did not become healthy." >&2
  tail -n 80 "$STDERR_LOG" >&2 || true
  return 1
}

stop_service() {
  if launchctl remove "$LABEL" 2>/dev/null; then
    echo "Eagle Eye stopped."
  else
    echo "Eagle Eye was not running."
  fi
}

show_status() {
  if launchctl print "$DOMAIN" >/dev/null 2>&1; then
    launchctl print "$DOMAIN" | awk '/state =|pid =|last exit code =/ {print}'
    if curl --silent --fail "$health_url" >/dev/null 2>&1; then
      echo "Health: ok ($health_url)"
    else
      echo "Health: unavailable ($health_url)"
      return 1
    fi
  else
    echo "Eagle Eye is not running."
    return 1
  fi
}

case "$ACTION" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    show_status
    ;;
  logs)
    tail -n 80 "$STDOUT_LOG" "$STDERR_LOG"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
