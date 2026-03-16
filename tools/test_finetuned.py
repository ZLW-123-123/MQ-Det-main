# Adapted from tools/test_grounding_net.py
import argparse
import os
import datetime
import torch
import torch.distributed as dist
import shutil

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
        timeout=datetime.timedelta(0, 7200)
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main():
    parser = argparse.ArgumentParser(description="Inference for Fine-tuned MQ-Det Model")
    parser.add_argument(
        "--config-file",
        default="configs/finetune/mq-groundingdino-t.yaml",
        metavar="FILE",
        help="path to base config file",
    )
    parser.add_argument(
        "--task_config",
        default="configs/test/dataset1.yaml",
        metavar="FILE",
        help="path to task config file",
    )
    parser.add_argument(
        "--additional_model_config",
        default="configs/vision_query_1shot/test.yaml",
        help="path to additional model config (e.g. for few-shot)",
    )
    parser.add_argument(
        "--weight",
        required=True,
        metavar="FILE",
        help="path to model weights (e.g. model_best.pth)",
    )
    parser.add_argument(
        "--query_bank_path",
        required=True,
        help="Path to Query Bank (custom/task specific .pth file)",
    )
    parser.add_argument(
        "--output_json_name",
        default="prediction.json",
        help="Filename for the output COCO format json (e.g. artaxor_1shot.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="OUTPUT/inference_results",
        help="Directory to save the results",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        init_distributed_mode(args)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    # Merge configs
    cfg.merge_from_file(args.config_file)
    if args.task_config:
        cfg.merge_from_file(args.task_config)
    if args.additional_model_config:
        cfg.merge_from_file(args.additional_model_config)
    cfg.merge_from_list(args.opts)

    # Explicitly set Query Bank Path
    if args.query_bank_path:
        cfg.VISION_QUERY.QUERY_BANK_PATH = args.query_bank_path
    
    cfg.freeze()

    # Create output directory
    mkdir(args.output_dir)

    logger = setup_logger("maskrcnn_benchmark", args.output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Task Config: {}".format(args.task_config))
    logger.info("Running inference with weights: {}".format(args.weight))
    logger.info("Query Bank: {}".format(args.query_bank_path))

    # Build Model
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Load Weights
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=args.output_dir)
    checkpointer.load(args.weight, force=True)

    # Inference Configuration
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    dataset_names = cfg.DATASETS.TEST
    if isinstance(dataset_names[0], (list, tuple)):
        dataset_names = [dataset for group in dataset_names for dataset in group]

    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=args.distributed)

    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        # Run inference
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY and (cfg.MODEL.RPN_ARCHITECTURE == "RPN" or cfg.DATASETS.CLASS_AGNOSTIC),
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=args.output_dir,
            cfg=cfg
        )
        synchronize()

        # Rename result file to target output_json_name
        if get_rank() == 0:
            # Note: maskrcnn_benchmark evaluation typically produces 'bbox.json' 
            # or '{dataset_name}_bbox_results.json' depending on implementation details in coco_eval.py
            # Based on standard libraries used here, it's often consistent.
            # We look for the most recently created .json file in the output dir that contains 'bbox' or matches typical patterns,
            # or specifically 'bbox.json' as defined in coco_eval.py (which we saw earlier).
            
            # Known behavior from earlier context: 
            # with open(json_result_file, "w") as f: json.dump(coco_results, f)
            # where json_result_file is passed from do_coco_evaluation usually as os.path.join(output_folder, "bbox.json")
            
            possible_src_files = [
                os.path.join(args.output_dir, "bbox.json"),
                os.path.join(args.output_dir, f"{dataset_name}_bbox_results.json")
            ]
            
            found = False
            for src_json in possible_src_files:
                if os.path.exists(src_json):
                    dst_json = os.path.join(args.output_dir, args.output_json_name)
                    try:
                        shutil.copy(src_json, dst_json)
                        logger.info(f"✅ Successfully saved prediction results to: {dst_json}")
                        found = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to rename result file: {e}")
            
            if not found:
                logger.warning(f"Could not find generated bbox.json file in {args.output_dir}. Please check output directory content.")
                # List files for debugging help
                files = os.listdir(args.output_dir)
                logger.info(f"Files in output dir: {files}")

if __name__ == "__main__":
    main()
