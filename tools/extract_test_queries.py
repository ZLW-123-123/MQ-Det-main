import argparse
import os
from pathlib import Path
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# 定义自定义数据集配置映射
custom_configs = {
    'dataset1': 'configs/test/dataset1.yaml',
    'dataset2': 'configs/test/dataset2.yaml',
    'dataset3': 'configs/test/dataset3.yaml',
}

def main():
    parser = argparse.ArgumentParser(description="Extract Vision Query for Custom Datasets")
    parser.add_argument("--python", default='python', type=str, help='python command')
    parser.add_argument(
        "--config-file",
        default="configs/finetune/mq-groundingdino-t.yaml",
        metavar="FILE",
        help="path to base config file",
        type=str,
    )

    parser.add_argument("--datasets", default='dataset1,dataset2,dataset3', type=str, help="dataset1,dataset2,dataset3 or all")

    parser.add_argument("--shots", default='1,5,10', type=str, help="1,5,10")
    
    parser.add_argument("--add_name", default="tiny", type=str)

    parser.add_argument("--output_root", default="MODEL/test_queries", type=str)
    parser.add_argument("--model_weight", default="OUTPUT/MQ-GD-COCO-Pretrain/model_final.pth", type=str)

    args = parser.parse_args()


    if args.datasets == 'all':
        target_datasets = list(custom_configs.keys())
    else:
        target_datasets = args.datasets.split(',')


    target_shots = [int(s) for s in args.shots.split(',')]

    mkdir(args.output_root)

    for dataset_name in target_datasets:
        if dataset_name not in custom_configs:
            print(f"Warning: Dataset {dataset_name} not found in custom_configs. Skipping.")
            continue
        
        task_config_path = custom_configs[dataset_name]
        
        for shot in target_shots:
            print(f"\n[Processing] Dataset: {dataset_name} | Shot: {shot}")
            
            save_name = '{}_query_{}_pool7_sel_{}.pth'.format(dataset_name, shot, args.add_name)
            save_path = str(Path(args.output_root, save_name))
            

            if os.path.exists(save_path):
                 print(f"Skipping {save_path}, already exists.")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cmd = (
                f"{args.python} tools/train_net_finetune.py "
                f"--config-file {args.config_file} "
                f"--task_config {task_config_path} "
                f"--additional_model_config configs/vision_query_{shot}shot/test.yaml "
                f"--extract_query "
                f"VISION_QUERY.QUERY_BANK_SAVE_PATH {save_path} "
                f"VISION_QUERY.MAX_QUERY_NUMBER {shot} "
                f"DATASETS.FEW_SHOT {shot} "
                f"VISION_QUERY.DATASET_NAME {dataset_name} "
                f"VISION_QUERY.QUERY_BANK_PATH '' "
                f"DATALOADER.NUM_WORKERS 4 "
                f"MODEL.WEIGHT {args.model_weight} "
                f"OUTPUT_DIR {args.output_root} "
                f"DATASETS.TRAIN \"('train_{shot}_3',)\" "
                f"DATASETS.TEST \"('test',)\" "
            )
            
            print(f"Running: {cmd}")
            ret = os.system(cmd)
            if ret != 0:
                print(f"Error executing command for {dataset_name} {shot}-shot")

if __name__ == "__main__":
    main()
