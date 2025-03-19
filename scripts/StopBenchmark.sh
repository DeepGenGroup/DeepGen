#!/bin/bash
# 查找所有包含 "Get" 关键字的进程（排除 grep 自身）
pids=$(pgrep -f "deepGenMain" -u $USER)

if [ -z "$pids" ]; then
    echo "未找到任何 deepGenMain 进程"
    exit 0
fi

# 遍历所有匹配的进程
for pid in $pids; do
    # 获取进程详细信息用于验证
    cmdline=$(ps -p $pid -o cmd=)
    
    # 排除误匹配（例如路径含 "Get" 的其他进程）
    if [[ ! $cmdline =~ .*Get.* ]]; then
        continue
    fi
    
    echo "发现目标进程: PID=$pid, CMD=$cmdline"
    
    # 获取进程组 ID（PGID）
    pgid=$(ps -o pgid= $pid | tr -d ' ')
    
    # 安全校验：禁止操作 PGID=1（init 进程）
    if [ "$pgid" -eq 1 ]; then
        echo "警告：跳过系统关键进程 (PGID=1)"
        continue
    fi
    
    # 杀死整个进程组（包括子进程）
    echo "终止进程组 PGID=$pgid..."
    
    # 先尝试优雅终止
    kill -TERM -- -$pgid 2>/dev/null
    
    # 等待 5 秒后检查残留
    sleep 5
    if ps -p $pid >/dev/null 2>&1; then
        echo "强制终止进程组 PGID=$pgid..."
        kill -9 -- -$pgid 2>/dev/null
    fi
done

# 二次确认清理结果
remaining=$(pgrep -f "deepGenMain")
if [ -n "$remaining" ]; then
    echo "警告：以下进程未被终止: $remaining"
else
    echo "所有 deepGenMain 进程及子进程已终止"
fi