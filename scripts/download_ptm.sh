#!/usr/bin/env bash
# 下载 PTM 数据集（版权归 CKCEST，仅限科研使用）到 data/raw/PTM
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ -d data/raw/PTM/.git ]; then
  echo "data/raw/PTM 已存在，跳过克隆"
else
  git clone --depth 1 https://github.com/yao8839836/PTM.git data/raw/PTM
fi
ls data/raw/PTM/data | head -40
echo "OK: PTM 数据位于 data/raw/PTM/data"
