#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
AUDIT_DIR="$PROJECT_ROOT/video_test/botsort_lv006/error_audit"
CONTACT_SHEET="$AUDIT_DIR/contact_sheet.html"

SERVER_BIND="127.0.0.1"
SERVER_PORT="${LV006_CONTACT_SHEET_PORT:-8765}"
WINDOWS_PORT="${LV006_CONTACT_SHEET_LOCAL_PORT:-8766}"
RUNTIME_BASE="${XDG_RUNTIME_DIR:-/tmp}"
RUNTIME_TAG="lv006_contact_sheet_$(id -u)_${SERVER_PORT}"
PID_FILE="$RUNTIME_BASE/${RUNTIME_TAG}.pid"
UNIT_NAME="lv006-contact-sheet-${SERVER_PORT}.service"

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

validate_port() {
    local label="$1"
    local value="$2"

    case "$value" in
        ''|*[!0-9]*) die "$label 必须是 1–65535 的整数，当前值：$value" ;;
    esac
    if ((value < 1 || value > 65535)); then
        die "$label 必须是 1–65535 的整数，当前值：$value"
    fi
}

is_expected_server_pid() {
    local pid="$1"
    local process_cwd
    local -a process_args=()

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    mapfile -d '' -t process_args < "/proc/$pid/cmdline" || return 1
    ((${#process_args[@]} >= 6)) || return 1
    [[ "${process_args[0]##*/}" == python3 ]] || return 1
    [[ "${process_args[1]}" == -m ]] || return 1
    [[ "${process_args[2]}" == http.server ]] || return 1
    [[ "${process_args[3]}" == "$SERVER_PORT" ]] || return 1
    [[ "${process_args[4]}" == --bind ]] || return 1
    [[ "${process_args[5]}" == "$SERVER_BIND" ]] || return 1
    process_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" || return 1
    [[ "$process_cwd" == "$AUDIT_DIR" ]]
}

managed_unit_pid() {
    local pid

    systemctl --user is-active --quiet "$UNIT_NAME" 2>/dev/null || return 1
    pid="$(systemctl --user show "$UNIT_NAME" --property=MainPID --value 2>/dev/null)" || return 1
    is_expected_server_pid "$pid" || return 1
    printf '%s\n' "$pid"
}

find_server_pid() {
    local pid

    if pid="$(managed_unit_pid)"; then
        printf '%s\n' "$pid"
        return 0
    fi

    if [[ -r "$PID_FILE" ]]; then
        read -r pid < "$PID_FILE" || true
        if is_expected_server_pid "${pid:-}"; then
            printf '%s\n' "$pid"
            return 0
        fi
    fi

    while read -r pid; do
        if is_expected_server_pid "$pid"; then
            printf '%s\n' "$pid"
            return 0
        fi
    done < <(pgrep -u "$(id -u)" -f "python3 -m http.server ${SERVER_PORT} --bind ${SERVER_BIND}" 2>/dev/null || true)

    return 1
}

listener_details() {
    ss -H -ltnp "sport = :$SERVER_PORT" 2>/dev/null || true
}

server_ip() {
    local address

    address="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')"
    if [[ -z "$address" ]]; then
        address="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi
    printf '%s\n' "${address:-<服务器IP>}"
}

print_windows_start_steps() {
    local address
    local remote_user

    address="$(server_ip)"
    remote_user="$(id -un)"
    cat <<EOF

Windows 本机需要配合：
  1. 打开 Windows PowerShell，不要在 Linux/Codex 集成终端执行。
  2. 建立 SSH 本地隧道，并保持该窗口运行：

     ssh -N -L ${WINDOWS_PORT}:127.0.0.1:${SERVER_PORT} ${remote_user}@${address}

  3. 在 Codex 内置浏览器打开：

     http://127.0.0.1:${WINDOWS_PORT}/contact_sheet.html

  4. 可在另一个 PowerShell 窗口验证：

     curl.exe -I http://127.0.0.1:${WINDOWS_PORT}/contact_sheet.html

如果出现“remote port forwarding failed for listen port 7890”，但 SSH 命令
仍停留在前台且没有返回 PowerShell 提示符，当前 ${WINDOWS_PORT} 隧道通常仍可使用。
若该警告导致 SSH 立即退出，可忽略 Windows SSH 配置后重试：

     ssh -F NUL -N -L ${WINDOWS_PORT}:127.0.0.1:${SERVER_PORT} ${remote_user}@${address}
EOF
}

print_windows_stop_steps() {
    cat <<EOF

Windows 本机需要配合：
  1. 切换到运行 SSH 隧道的 PowerShell 窗口，按 Ctrl+C。
  2. 关闭 contact_sheet.html 浏览器标签页。
EOF
}

start_server() {
    local pid
    local listener
    local attempt
    local python_bin
    local systemd_output

    [[ -d "$AUDIT_DIR" ]] || die "审核目录不存在：$AUDIT_DIR"
    [[ -f "$CONTACT_SHEET" ]] || die "联系表不存在：$CONTACT_SHEET"

    if grep -q 'file:///' "$CONTACT_SHEET"; then
        die "contact_sheet.html 仍包含 file:// 链接；请先改成 cases/ 和 thumbnails/ 相对路径。"
    fi

    if pid="$(find_server_pid)"; then
        printf 'LV006 联系表服务已经运行，无需重复启动。\n'
        printf '远端地址：http://%s:%s/contact_sheet.html\n' "$SERVER_BIND" "$SERVER_PORT"
        printf '进程 PID：%s\n' "$pid"
        print_windows_start_steps
        return 0
    fi

    listener="$(listener_details)"
    if [[ -n "$listener" ]]; then
        printf '端口 %s 已被其他程序占用，出于安全考虑不会启动或终止它：\n%s\n' "$SERVER_PORT" "$listener" >&2
        return 1
    fi

    systemctl --user show-environment >/dev/null 2>&1 || die "当前用户的 systemd 服务管理器不可用"
    python_bin="$(command -v python3)" || die "找不到 python3"
    systemctl --user reset-failed "$UNIT_NAME" >/dev/null 2>&1 || true
    if ! systemd_output="$(
        systemd-run \
            --user \
            --unit="$UNIT_NAME" \
            --collect \
            --working-directory="$AUDIT_DIR" \
            "$python_bin" -m http.server "$SERVER_PORT" --bind "$SERVER_BIND" 2>&1
    )"; then
        die "无法创建用户级 systemd 网页服务：$UNIT_NAME；$systemd_output"
    fi

    for attempt in {1..30}; do
        if curl --noproxy '*' -fs -o /dev/null "http://${SERVER_BIND}:${SERVER_PORT}/contact_sheet.html"; then
            break
        fi
        sleep 0.1
    done

    if ! curl --noproxy '*' -fsS -o /dev/null "http://${SERVER_BIND}:${SERVER_PORT}/contact_sheet.html"; then
        systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
        rm -f -- "$PID_FILE"
        printf '网页服务启动后未通过 HTTP 检查。最近日志：\n' >&2
        journalctl --user -u "$UNIT_NAME" --no-pager -n 20 >&2 || true
        return 1
    fi

    pid="$(find_server_pid)" || die "服务已响应，但无法确认其进程"
    printf '已启动 LV006 联系表服务。\n'
    printf '远端地址：http://%s:%s/contact_sheet.html\n' "$SERVER_BIND" "$SERVER_PORT"
    printf '进程 PID：%s\n' "$pid"
    printf '日志命令：journalctl --user -u %s --no-pager\n' "$UNIT_NAME"
    print_windows_start_steps
}

stop_server() {
    local pid
    local attempt

    if ! pid="$(find_server_pid)"; then
        rm -f -- "$PID_FILE"
        printf 'LV006 联系表服务当前没有运行。\n'
        if [[ -n "$(listener_details)" ]]; then
            printf '注意：端口 %s 被其他程序占用，本脚本没有终止它。\n' "$SERVER_PORT"
        fi
        print_windows_stop_steps
        return 0
    fi

    if [[ "$(managed_unit_pid 2>/dev/null || true)" == "$pid" ]]; then
        systemctl --user stop "$UNIT_NAME" || die "无法停止用户级 systemd 服务：$UNIT_NAME"
    else
        kill "$pid" || die "无法向 PID $pid 发送正常终止信号"
    fi
    for attempt in {1..50}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    if kill -0 "$pid" 2>/dev/null; then
        printf 'PID %s 未在 5 秒内退出；本脚本不会自动使用 kill -9。\n' "$pid" >&2
        printf '请先检查：ps -p %s -o pid=,args=\n' "$pid" >&2
        return 1
    fi

    rm -f -- "$PID_FILE"
    printf '已关闭 LV006 联系表服务（原 PID：%s）。\n' "$pid"
    print_windows_stop_steps
}

show_status() {
    local pid

    if pid="$(find_server_pid)"; then
        printf '状态：运行中\n'
        printf '远端地址：http://%s:%s/contact_sheet.html\n' "$SERVER_BIND" "$SERVER_PORT"
        printf '进程 PID：%s\n' "$pid"
        if curl --noproxy '*' -fsS -o /dev/null "http://${SERVER_BIND}:${SERVER_PORT}/contact_sheet.html"; then
            printf 'HTTP 检查：通过\n'
        else
            printf 'HTTP 检查：失败，请查看日志或进程状态\n' >&2
        fi
        print_windows_start_steps
    else
        printf '状态：未运行\n'
        if [[ -n "$(listener_details)" ]]; then
            printf '注意：端口 %s 被其他程序占用。\n' "$SERVER_PORT"
            listener_details
        fi
        printf '运行本脚本（不带参数）或使用 start 即可启动。\n'
    fi
}

show_usage() {
    cat <<EOF
用法：
  $(basename "$0")          自动切换：运行中则关闭，未运行则启动
  $(basename "$0") start    启动；已运行时只显示状态和本地步骤
  $(basename "$0") stop     关闭；未运行时安全返回
  $(basename "$0") status   只读检查状态
  $(basename "$0") restart  重启

可选环境变量：
  LV006_CONTACT_SHEET_PORT        远端服务端口，默认 8765
  LV006_CONTACT_SHEET_LOCAL_PORT  Windows 本地隧道端口，默认 8766
EOF
}

validate_port "远端服务端口" "$SERVER_PORT"
validate_port "Windows 本地隧道端口" "$WINDOWS_PORT"

ACTION="${1:-toggle}"
case "$ACTION" in
    toggle)
        if find_server_pid >/dev/null; then
            stop_server
        else
            start_server
        fi
        ;;
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        show_status
        ;;
    restart)
        stop_server && start_server
        ;;
    -h|--help|help)
        show_usage
        ;;
    *)
        show_usage >&2
        exit 2
        ;;
esac
