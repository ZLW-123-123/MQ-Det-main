# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, all_gather, is_main_process, broadcast_data, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.ema import ModelEma
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from .inference import inference
import pdb

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_data_loader=None,
        meters=None,
        zero_shot=False,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    # meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.SOLVER.MODEL_EMA)
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) // cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(data_loader) // cfg.SOLVER.MAX_EPOCH )

    if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
        patience_counter = 0
        previous_best = 0.0

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1
                
    burn_in_iter = int(max_iter * 0.3)
    dynamic_thresholds = {}
    from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
            if positive_map is not None:
                positive_map = positive_map.to(device)
            if positive_map_eval is not None:
                positive_map_eval = positive_map_eval.to(device)
        except:
            pass
            
        if positive_map_eval is None and positive_map is not None:
            if isinstance(positive_map, torch.Tensor):
                # Construct dict exactly like inference.py or GroundingDINO eval
                positive_map_eval_dict = {}
                for tgt in targets:
                    if tgt.has_field("labels_in_caption") and tgt.has_field("all_map"):
                        lbls = tgt.get_field("labels_in_caption")
                        pm_tensors = tgt.get_field("all_map")
                        for lbl_val, pm_t in zip(lbls, pm_tensors):
                            if isinstance(lbl_val, torch.Tensor):
                                lbl_val = lbl_val.item()
                            tokens = torch.nonzero(pm_t, as_tuple=True)[0].tolist()
                            if lbl_val not in positive_map_eval_dict:
                                positive_map_eval_dict[lbl_val] = tokens
                    elif tgt.has_field("labels") and tgt.has_field("positive_map"):
                        lbls = tgt.get_field("labels")
                        pm_tensors = tgt.get_field("positive_map")
                        for lbl_tensor, pm_t in zip(lbls, pm_tensors):
                            lbl_val = lbl_tensor.item()
                            tokens = torch.nonzero(pm_t, as_tuple=True)[0].tolist()
                            if lbl_val not in positive_map_eval_dict:
                                positive_map_eval_dict[lbl_val] = tokens
                positive_map_eval = positive_map_eval_dict
            else:
                import copy
                positive_map_eval = copy.deepcopy(positive_map)

        # Freeze language backbone
        if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            if hasattr(model, "module"):
                model.module.language_backbone.eval()
            else:
                model.language_backbone.eval()

        if iteration > burn_in_iter and model_ema is not None:
            model_ema.ema.eval()
            with torch.no_grad():
                if cfg.SOLVER.USE_AMP:
                    with autocast():
                        if captions is not None and len(captions) > 0:
                            pseudo_outputs = model_ema.ema(images, captions=captions, positive_map=positive_map_eval)
                        else:
                            pseudo_outputs = model_ema.ema(images)
                else:
                    if captions is not None and len(captions) > 0:
                        pseudo_outputs = model_ema.ema(images, captions=captions, positive_map=positive_map_eval)
                    else:
                        pseudo_outputs = model_ema.ema(images)
            
            merged_targets = []
            for t_gt, p_out in zip(targets, pseudo_outputs):
                if len(p_out) == 0:
                    merged_targets.append(t_gt)
                    continue
                p_labels = p_out.get_field("labels")
                p_scores = p_out.get_field("scores")
                keep_mask = torch.zeros_like(p_scores, dtype=torch.bool)
                unique_labels = p_labels.unique()
                
                # 获取该图原本存在的GT类别
                gt_labels_in_image = t_gt.get_field("labels").unique().tolist() if t_gt.has_field("labels") else []
                
                for label in unique_labels:
                    lbl_idx = label.item()
                    if lbl_idx not in dynamic_thresholds:
                        dynamic_thresholds[lbl_idx] = 0.5
                    cls_mask = p_labels == label
                    cls_scores = p_scores[cls_mask]
                    
                    # Compute mean only from somewhat confident bounding boxes to prevent threshold from pluning to 0.01
                    top_scores = cls_scores[cls_scores > 0.15]
                    if len(top_scores) > 0:
                        dynamic_thresholds[lbl_idx] = 0.9 * dynamic_thresholds[lbl_idx] + 0.1 * top_scores.mean().item()
                    
                    # STRATEGY 1: 双重阈值（Dual-Thresholding）
                    if lbl_idx in gt_labels_in_image:
                        # 对于图里原本就有的类别，门槛放宽，尽可能挖掘更多共生的小目标
                        cur_thresh = max(0.20, dynamic_thresholds[lbl_idx] * 0.6)
                    else:
                        # 对于这图里没见过的类别，很可能是视觉语言模型的背景幻觉，必须加严审核
                        cur_thresh = max(0.40, dynamic_thresholds[lbl_idx] * 0.8)
                        
                    keep_mask[cls_mask] = cls_scores >= cur_thresh
                    
                p_out = p_out[keep_mask]
                
                # STRATEGY 2: 巨型畸形框过滤（Size/Area Heuristic）
                # 舍弃占位面积超过全图 80% 的框，它们绝大多数是把水体、大礁石等认成了特定物体
                if len(p_out) > 0:
                    img_w, img_h = p_out.size
                    img_area = img_w * img_h
                    p_out_area = p_out.area()
                    
                    area_mask = p_out_area <= (0.80 * img_area)
                    p_out = p_out[area_mask]
                
                # --- APPLY NMS to Pseudo Labels ---
                if len(p_out) > 0:
                    from torchvision.ops import batched_nms
                    # convert to xyxy before NMS
                    p_out_xyxy = p_out.convert("xyxy")
                    # 使用 0.45 可以保留更多局部特征的框，之前 0.25 限制得太狠导致重叠目标被误杀
                    keep_nms = batched_nms(p_out_xyxy.bbox, p_out.get_field("scores"), p_out.get_field("labels"), iou_threshold=0.45)
                    p_out = p_out[keep_nms]
                
                # Match mode and size to Ground Truth (CRITICAL for correct box concatenation)
                if len(p_out) > 0 and len(t_gt) > 0:
                    p_out = p_out.resize(t_gt.size)
                    p_out = p_out.convert(t_gt.mode)
                    
                    ious = boxlist_iou(p_out, t_gt)
                    max_iou, _ = ious.max(dim=1)
                    keep_iou = max_iou < 0.3
                    p_out = p_out[keep_iou]
                
                # Verify that p_labels can actually be mapped to text tokens, drop invalid labels
                if len(p_out) > 0 and t_gt.has_field("positive_map"):
                    labels_in_caption = t_gt.get_field("labels_in_caption") if t_gt.has_field("labels_in_caption") else []
                    all_map = t_gt.get_field("all_map") if t_gt.has_field("all_map") else None
                    p_labels = p_out.get_field("labels")
                    
                    valid_mask = torch.zeros(len(p_out), dtype=torch.bool, device=p_labels.device)
                    for idx_plbl, lbl in enumerate(p_labels):
                        lbl_val = lbl.item()
                        if isinstance(positive_map_eval, dict) and lbl_val in positive_map_eval:
                            valid_mask[idx_plbl] = True
                        elif all_map is not None and lbl_val in labels_in_caption:
                            valid_mask[idx_plbl] = True
                            
                    p_out = p_out[valid_mask]
                    
                if len(p_out) > 0:
                    from maskrcnn_benchmark.structures.bounding_box import BoxList
                    merged_boxes = torch.cat((t_gt.bbox, p_out.bbox), dim=0)
                    merged_tgt = BoxList(merged_boxes, t_gt.size, mode=t_gt.mode)
                    
                    labels_in_caption = t_gt.get_field("labels_in_caption") if t_gt.has_field("labels_in_caption") else []
                    all_map = t_gt.get_field("all_map") if t_gt.has_field("all_map") else None
                    p_labels = p_out.get_field("labels")
                    
                    new_pos_map_rows = []
                    if t_gt.has_field("positive_map"):
                        max_len = t_gt.get_field("positive_map").shape[1]
                        for lbl in p_labels:
                            lbl_val = lbl.item()
                            row = torch.zeros(max_len, dtype=t_gt.get_field("positive_map").dtype, device=t_gt.get_field("positive_map").device)
                            
                            # MUST NOT use elif! Both could be true, but positive_map_eval is more complete!
                            if isinstance(positive_map_eval, dict) and lbl_val in positive_map_eval:
                                num_tokens = len(positive_map_eval[lbl_val])
                                for t_idx in positive_map_eval[lbl_val]:
                                    if t_idx < max_len:
                                        row[t_idx] = 1.0 / num_tokens if num_tokens > 0 else 1.0
                            elif all_map is not None and lbl_val in labels_in_caption:
                                idx = labels_in_caption.index(lbl_val)
                                row = all_map[idx].clone().to(row.device)
                                
                            new_pos_map_rows.append(row)
                                
                    for f in t_gt.fields():
                        if f == "labels":
                            merged_tgt.add_field("labels", torch.cat((t_gt.get_field("labels"), p_labels), dim=0))
                        elif f == "positive_map":
                            if len(new_pos_map_rows) > 0:
                                new_pos_map = torch.stack(new_pos_map_rows, dim=0).to(t_gt.get_field("positive_map").device)
                                old_pos_map = t_gt.get_field("positive_map")
                                merged_pos_map = torch.cat([old_pos_map, new_pos_map], dim=0)
                                merged_tgt.add_field("positive_map", merged_pos_map)
                            else:
                                merged_tgt.add_field("positive_map", t_gt.get_field("positive_map"))
                        elif f in ["normed_cxcy_boxes", "masks", "is_box_mask", "scores", "tokens_positive"]:
                            pass # Skip per-box attributes we can't easily merge and don't need for GroundingDINO
                        else:
                            merged_tgt.add_field(f, t_gt.get_field(f))
                            
                    # Recompute normed_cxcy_boxes for the merged target
                    W, H = merged_tgt.size
                    merged_bbox = merged_tgt.bbox
                    normed_bbox = merged_bbox / torch.tensor([[W, H, W, H]], dtype=torch.float32, device=merged_bbox.device)
                    x0, y0, x1, y1 = normed_bbox.unbind(-1)
                    normed_cxcy_boxes = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)
                    merged_tgt.add_field("normed_cxcy_boxes", normed_cxcy_boxes)
                    
                    t_gt = merged_tgt
                merged_targets.append(t_gt)

            targets = merged_targets

            if len(targets) > 0 and "positive_map" in targets[0].fields():
                max_len = max([v.get_field("positive_map").shape[1] for v in targets])
                nb_boxes = sum([v.get_field("positive_map").shape[0] for v in targets])
                batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=targets[0].get_field("positive_map").dtype, device=targets[0].get_field("positive_map").device)
                cur_count = 0
                for v in targets:
                    cur_pos = v.get_field("positive_map")
                    batched_pos_map[cur_count: cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
                    cur_count += len(cur_pos)
                positive_map = batched_pos_map.float()

        if cfg.SOLVER.USE_AMP:
            with autocast():
                if len(captions) > 0:
                    loss_dict = model(images, targets, captions=captions, positive_map=positive_map, greenlight_map = greenlight_map)
                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # save checkpoints for further debug if nan happens
            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     logging.error("Losses are : {}".format(loss_dict))
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #             dict_to_save,
            #             fname
            #         )


            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # loss_value = losses.item()
            # if not math.isfinite(loss_value):
            #     logging.error(f'=> loss is {loss_value}, stopping training')
            #     time_str = time.strftime('%Y-%m-%d-%H-%M')
            #     fname = os.path.join(checkpointer.save_dir, f'{time_str}_states.pth')
            #     logging.info(f'=> save error state to {fname}')
            #     dict_to_save = {
            #         'x': images,
            #         'y': targets,
            #         'loss': losses,
            #         'states': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            #     }
            #     if len(captions) > 0:
            #         dict_to_save['captions'] = captions
            #         dict_to_save['positive_map'] = positive_map
            #     torch.save(
            #         dict_to_save,
            #         fname
            #     )
                

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float('inf')
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if 'weight_decay' in param:
                        param['weight_decay'] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
        # if iteration % 1 == 0 or iteration == max_iter:
            #logger.info(
            if global_rank <= 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
            if is_main_process():
                logger.info("Evaluating")
            eval_result = 0.0
            model.eval()

            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False,
                        disable_print=True
                    )
                    if is_main_process():
                        eval_result = _result[0].results['bbox']['AP']
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = all_gather(results_dict)
                if is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                              box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = all_gather(eval_result)[0] #broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)
            
            if cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1:
                if eval_result < previous_best:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    previous_best = eval_result
                    checkpointer.save("model_best", **arguments)
                # print("Previous Best", previous_best, "Patience Counter", patience_counter, "Eval Result", eval_result)
                if is_main_process():
                    logger.info("Previous Best {} Patience Counter {} Eval Result {}".format(previous_best, patience_counter, eval_result))
                if patience_counter >= cfg.SOLVER.AUTO_TERMINATE_PATIENCE:
                    if is_main_process():
                        # print("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                        logger.info("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(iteration, previous_best))
                    break

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
