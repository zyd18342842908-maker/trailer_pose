#!/usr/bin/env bash
# Configure this workspace's origin without staging, committing or pushing files.
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
default_url='https://github.com/zyd18342842908-maker/trailer_pose.git'

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
    printf '用法：bash scripts/connect_github.sh [GitHub仓库URL]\n默认仓库：%s\n仅设置本地 origin，不自动提交或上传文件。\n' "$default_url"
    exit 0
fi
if (( $# > 1 )); then
    printf '仅接受一个仓库地址；使用 --help 查看用法。\n' >&2
    exit 2
fi
repository_url="${1:-$default_url}"
case "$repository_url" in
    https://github.com/*/*|git@github.com:*/*|ssh://git@github.com/*/*) ;;
    *) printf '请输入 GitHub 的 HTTPS 或 SSH 仓库地址。\n' >&2; exit 2 ;;
esac

if ! command -v git >/dev/null 2>&1; then
    printf '没有找到 git。Ubuntu 上可先运行 sudo apt install git，再重新执行本脚本。\n' >&2
    exit 127
fi

existing_root="$(git -C "$project_root" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -n "$existing_root" ]]; then
    existing_root="$(CDPATH= cd -- "$existing_root" && pwd -P)"
    if [[ "$existing_root" != "$project_root" ]]; then
        printf '当前工程属于上级 Git 仓库；未修改其远程配置。\n' >&2
        exit 1
    fi
else
    if [[ -e "$project_root/.git" && ! -w "$project_root/.git" ]]; then
        printf '工程的 .git 目录不可写，请在拥有该工程写权限的本机终端执行。\n' >&2
        exit 1
    fi
    git -C "$project_root" init
    git -C "$project_root" symbolic-ref HEAD refs/heads/main
fi

normalize_url() {
    local value="${1%/}"
    value="${value%.git}"
    value="${value#https://github.com/}"
    value="${value#git@github.com:}"
    value="${value#ssh://git@github.com/}"
    printf '%s' "$value"
}

if current_url="$(git -C "$project_root" remote get-url origin 2>/dev/null)"; then
    if [[ "$(normalize_url "$current_url")" != "$(normalize_url "$repository_url")" ]]; then
        printf 'origin 已指向另一个仓库；未覆盖，请先检查现有远程配置。\n' >&2
        exit 1
    fi
else
    git -C "$project_root" remote add origin "$repository_url"
fi

verified_url="$(git -C "$project_root" remote get-url origin)"
if [[ "$(normalize_url "$verified_url")" != "$(normalize_url "$repository_url")" ]]; then
    printf '关联校验失败，请检查 origin 配置。\n' >&2
    exit 1
fi
printf '已关联 GitHub 仓库：%s\n工程目录：%s\n' "$repository_url" "$project_root"
