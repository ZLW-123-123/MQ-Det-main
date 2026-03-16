#!/bin/bash

# source activate wwllzz
 
export WORLD_SIZE=1

# ================== 核心配置区域 ==================
# 定义要遍历的数据集和Shot数 (需与训练时保持一致)
DATASET_LIST=("dataset1" "dataset2" "dataset3")
SHOT_LIST=(1 5 10)
# ==========================================================

# 配置文件路径
CONFIG_FILE="configs/finetune/mq-groundingdino-t.yaml"

# 统一的推理结果输出目录
FINAL_OUTPUT_DIR="OUTPUT/Inference_Results"
mkdir -p ${FINAL_OUTPUT_DIR}

for DATASET_NAME in "${DATASET_LIST[@]}"; do
    for SHOT_NUM in "${SHOT_LIST[@]}"; do
        echo "======================================================================="
        echo "Start Inference: Dataset: ${DATASET_NAME}, Shot: ${SHOT_NUM}"
        echo "======================================================================="

        # 配置文件路径推导
        TASK_CONFIG="configs/test/${DATASET_NAME}.yaml"
        ADDITIONAL_MODEL_CONFIG="configs/vision_query_${SHOT_NUM}shot/test.yaml"
        
        # Query Bank 路径 (需与训练时一致)
        QUERY_BANK_PATH="MODEL/test_queries/${DATASET_NAME}_query_${SHOT_NUM}_pool7_sel_tiny.pth"
        
        # 训练好的模型路径 (根据 run_stage2_all_tasks.sh 的 pattern)
        TRAINED_DIR="OUTPUT/MQ-GD-${DATASET_NAME}-Finetune-${SHOT_NUM}shot"
        WEIGHT_PATH="${TRAINED_DIR}/model_final.pth"
        
        # 结果文件名
        OUTPUT_JSON_NAME="${DATASET_NAME}_${SHOT_NUM}shot.json"

        # 检查模型权重是否存在
        if [ ! -f "$WEIGHT_PATH" ]; then
            echo "[Warning] Model weights not found at: $WEIGHT_PATH"
            echo "Skipping ${DATASET_NAME} ${SHOT_NUM}-shot inference..."
            continue
        fi

        # 运行推理脚本
        # 注意: output_dir 设置为 FINAL_OUTPUT_DIR，结果 json 会生成在其中
        # 日志也会生成在该目录，多次运行可能会追加或覆盖 log.txt
        python tools/test_finetuned.py \
            --config-file ${CONFIG_FILE} \
            --task_config ${TASK_CONFIG} \
            --additional_model_config ${ADDITIONAL_MODEL_CONFIG} \
            --weight ${WEIGHT_PATH} \
            --query_bank_path ${QUERY_BANK_PATH} \
            --output_json_name ${OUTPUT_JSON_NAME} \
            --output_dir ${FINAL_OUTPUT_DIR} \
            TEST.IMS_PER_BATCH 1 \
            DATASETS.TEST "('test',)"
            
        echo "Finished inference for ${DATASET_NAME} ${SHOT_NUM}-shot. Result should be in ${FINAL_OUTPUT_DIR}/${OUTPUT_JSON_NAME}"
    done
done
