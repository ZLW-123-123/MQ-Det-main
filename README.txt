======================================================================
MQ-Det 项目运行指南
======================================================================

0. 环境准备 (Environment Setup)
----------------------------------------------------------------------
在运行代码之前，请先准备好 Python 环境。建议使用 Anaconda 或 Miniconda 创建一个新的虚拟环境，建议 Python 版本 3.9。

a) 创建并激活虚拟环境 (可选，如果已有环境可跳过)
   $ conda create -n mq-det python=3.9 -y
   $ conda activate mq-det

b) 安装依赖库
   项目根目录下提供了 `requirements.txt` 和 `init.sh`。
   请依次运行以下命令安装所需依赖：
   
   $ bash init.sh

   该脚本会自动安装所需依赖并编译相关扩展。

1. 数据集准备 (Datasets)
----------------------------------------------------------------------
项目配置文件默认使用相对路径 `datasets/` 来寻找数据。
请确保在项目根目录下创建一个名为 `datasets` 的文件夹，并将您的数据集（如 dataset1, dataset2, dataset3）放置在其中。

目录结构示例：
项目根目录/
  ├── configs/
  ├── tools/
  ├── ...
  └── datasets/
      ├── dataset1/
      │   ├── annotations/
      │   ├── train/
      │   └── test/
      ├── dataset2/
      └── dataset3/

如果您的数据集存放在其他位置，请通过创建软链接的方式将其映射到 `datasets` 目录：
$ ln -s /path/to/your/real/datasets ./datasets

2. 预训练模型 (Pretrained Model)
----------------------------------------------------------------------
请确保预训练权重文件放置在以下路径（相对路径）：
OUTPUT/MQ-GD-COCO-Pretrain/model_final.pth

如果没有该文件夹，请手动创建并将模型文件复制进去。

3. BERT 语言模型 (如果需要离线运行)
----------------------------------------------------------------------
配置文件 `configs/finetune/mq-groundingdino-t.yaml` 默认使用 HuggingFace 在线下载的 "bert-base-uncased"。
如果新环境无法联网，您需要：
1. 手动下载 "bert-base-uncased" 模型文件（config.json, pytorch_model.bin, vocab.txt 等）。
2. 将其放置在本地某个目录，例如 `pretrained_models/bert-base-uncased`。
3. 修改 `configs/finetune/mq-groundingdino-t.yaml` 文件：
   将 `TOKENIZER_TYPE`, `MODEL_TYPE` 和 `text_encoder_type` 的值从 "bert-base-uncased" 修改为您的本地绝对路径或相对路径。

4. Query Bank 文件
----------------------------------------------------------------------
运行 Stage 2 微调前，需要先提取 Query Bank。
确保运行 `run_extract_test_queries.sh`。该脚本已被配置为将结果输出到 `MODEL/test_queries/` 目录。
后续的 `run_stage2_all_test_tasks.sh` 会自动从该目录读取对应的 .pth 文件。

注意：配置文件 `configs/vision_query_Xshot/test.yaml` 中可能保留了一个默认的 `QUERY_BANK_PATH`，但在运行 Shell 脚本时，该参数会被脚本中动态生成的路径覆盖，因此无需手动修改配置文件中的这一项。

5. 运行微调代码 (Fine-tuning)
----------------------------------------------------------------------
执行命令：
$ bash run_stage2_all_test_tasks.sh

该脚本会自动遍历所有配置的数据集和 Shot 数（1/5/10），进行模型微调。
结果输出：
微调后的模型权重将保存在 `OUTPUT/` 目录下，按照以下命名规则生成子目录：
OUTPUT/MQ-GD-{DATASET_NAME}-Finetune-{SHOT_NUM}shot/
每个目录下的 `model_final.pth` 即为该任务微调后的最终模型文件。

6. 运行推理代码 (Inference)
----------------------------------------------------------------------
执行命令：
$ bash run_inference_all_tasks.sh

该脚本会加载第 5 步生成的微调模型，对测试集进行推理。
结果输出：
推理生成的 JSON 结果文件将保存在 `OUTPUT/Inference_Results/` 目录下。
文件命名格式为：{DATASET_NAME}_{SHOT_NUM}shot.json (例如 dataset1_1shot.json)。

