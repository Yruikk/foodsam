#!/bin/bash

# 检查是否安装了 conda
if ! command -v conda &> /dev/null; then
    echo "Conda 未安装，请先安装 Conda 后再运行此脚本。"
    exit 1
fi

#回到主文件夹下
# cd ..

# 创建 Conda 虚拟环境
echo "创建 Conda 虚拟环境 'foodsam2'..."
conda create -n foodsam2 python=3.11 -y

# 激活虚拟环境
echo "激活虚拟环境 'foodsam2'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate foodsam2

# 安装 PyTorch 和相关库
echo "安装 PyTorch、TorchVision 和 Torchaudio..."
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/

# 安装 mmcv
echo "安装 mmcv..."
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# 克隆并安装 sam2
echo "克隆并安装 sam2..."
if [ ! -d "sam2" ]; then
    git clone git@github.com:facebookresearch/sam2.git
else
    echo "sam2 已存在，跳过克隆。"
fi
cd sam2
pip install -e .
cd ..

# 检查并安装其他依赖项
echo "检查并安装其他依赖项..."
REQUIREMENTS_FILE="requirement.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    while read -r line; do
        PACKAGE=$(echo $line | awk '{print $1}')
        # 检查是否已安装
        if pip show $PACKAGE &> /dev/null; then
            echo "# $line 已安装，跳过。" >> temp_requirements.txt
        else
            echo "$line" >> temp_requirements.txt
        fi
    done < "$REQUIREMENTS_FILE"
    # 安装未安装的包
    if [ -f "temp_requirements.txt" ]; then
        pip install -r temp_requirements.txt
        rm temp_requirements.txt
    fi
else
    echo "未找到 requirement.txt，跳过依赖项安装。"
fi

pip install -r yolo_requirements.txt

echo "环境配置完成！您可以通过 'conda activate foodsam2' 激活环境。"