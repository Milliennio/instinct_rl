#!/bin/bash

# 1. 激活 Conda 环境
# 注意：你需要确保 source 命令指向你服务器上正确的 conda.sh 位置
# 常见的路径是 ~/anaconda3/etc/profile.d/conda.sh 或 ~/miniconda3/etc/profile.d/conda.sh
# 如果脚本运行提示找不到 source，请检查你的 conda 安装路径
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null

# 激活指定的环境
conda activate isaaclab

# 2. 使用绝对路径运行训练指令
# 不切换目录，直接执行
python /root/instinct_rl-main/InstinctLab-main/scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0