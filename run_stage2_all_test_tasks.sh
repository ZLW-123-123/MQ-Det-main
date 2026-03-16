#!/bin/bash

# source activate wwllzz
 
export WORLD_SIZE=1

# ================== 核心配置区域 (修改这里) ==================
# 定义要遍历的数据集和Shot数
DATASET_LIST=("dataset1" "dataset2" "dataset3")
SHOT_LIST=(1 5 10)
SHOT_SEED=3                     # 随机种子后缀 (默认3, 对应 train_X_3)
# ==========================================================

# 配置文件路径 (Stage 2 使用相同的 config，但通过 task_config 和 additional_model_config 调整)
CONFIG_FILE="configs/finetune/mq-groundingdino-t.yaml"

# Stage 1 预训练好的模型权重
PRETRAIN_MODEL="OUTPUT/MQ-GD-COCO-Pretrain/model_final.pth"

for DATASET_NAME in "${DATASET_LIST[@]}"; do
    for SHOT_NUM in "${SHOT_LIST[@]}"; do
        echo "======================================================================="
        echo "Start Finetuning Dataset: ${DATASET_NAME}, Shot: ${SHOT_NUM}"
        echo "======================================================================="

        # 任务配置 - 自动根据 DATASET_NAME 推导
        TASK_CONFIG="configs/test/${DATASET_NAME}.yaml"

        # 额外模型配置 - 自动根据 SHOT_NUM 推导
        ADDITIONAL_MODEL_CONFIG="configs/vision_query_${SHOT_NUM}shot/test.yaml"

        # Vision Query Bank 路径 - 自动推导
        QUERY_BANK_PATH="MODEL/test_queries/${DATASET_NAME}_query_${SHOT_NUM}_pool7_sel_tiny.pth"

        # Stage 2 输出目录 - 自动推导，避免覆盖
        OUTPUT_DIR="OUTPUT/MQ-GD-${DATASET_NAME}-Finetune-${SHOT_NUM}shot"

        # 构造训练集名称参数 (覆盖 YAML 中的 TRAIN)
        # 格式例如: "('train_1_3',)"
        TRAIN_DATASET_TUPLE="('train_${SHOT_NUM}_${SHOT_SEED}',)"
        TEST_DATASET_TUPLE="('test',)"

        # 启动训练 (Stage 2: Finetuning)
        # 直接使用 python 运行，不使用 distributed.launch，单卡模式下不需要 process group 和端口
        # --task_config: 指定具体的下游任务配置
        # --additional_model_config: 指定 Vision Query配置
        # --tuning_highlevel_override: 启用 vision_query设置 (冻结backbone等)
        # MODEL.WEIGHT: 加载预训练权重
        # DATASETS.TRAIN: 覆盖训练集配置

        python tools/train_net_finetune.py \
            --config-file ${CONFIG_FILE} \
            --task_config ${TASK_CONFIG} \
            --additional_model_config ${ADDITIONAL_MODEL_CONFIG} \
            --override_output_dir ${OUTPUT_DIR} \
            MODEL.WEIGHT ${PRETRAIN_MODEL} \
            SOLVER.IMS_PER_BATCH 1 \
            SOLVER.TUNING_HIGHLEVEL_OVERRIDE "vision_query" \
            VISION_QUERY.QUERY_BANK_PATH "${QUERY_BANK_PATH}" \
            DATASETS.TRAIN "${TRAIN_DATASET_TUPLE}" \
            DATASETS.TEST "${TEST_DATASET_TUPLE}" \
            TEST.DURING_TRAINING False \
            SOLVER.USE_AUTOSTEP False \
            SOLVER.MAX_EPOCH 12 \
            SOLVER.MODEL_EMA 0.999
            
    done
done
