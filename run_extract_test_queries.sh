#!/bin/bash
set -euo pipefail

# source activate wwllzz

# 脚本路径
PYTHON_SCRIPT="tools/extract_test_queries.py"

# 配置参数
DATASETS="dataset1,dataset2,dataset3" # 或者 指定 "ArTaxOr,clipart1k,FISH"
SHOTS="1,5,10"
OUTPUT_ROOT="MODEL/test_queries"
WEIGHT="OUTPUT/MQ-GD-COCO-Pretrain/model_final.pth"

echo "--------------------------------------------------------"
echo "Starting Custom Query Extraction..."
echo "Datasets: ${DATASETS}"
echo "Shots: ${SHOTS}"
echo "Output: ${OUTPUT_ROOT}"
echo "--------------------------------------------------------"

python "${PYTHON_SCRIPT}" \
    --datasets "${DATASETS}" \
    --shots "${SHOTS}" \
    --output_root "${OUTPUT_ROOT}" \
    --model_weight "${WEIGHT}"

echo "--------------------------------------------------------"
echo "Job Done! Check ${OUTPUT_ROOT} for .pth files."
echo "--------------------------------------------------------"
