# Adapted from https://github.com/IDEA-Research/GroundingDINO. The original liscenses are:
    # ------------------------------------------------------------------------
    # Grounding DINO
    # url: https://github.com/IDEA-Research/GroundingDINO
    # Copyright (c) 2023 IDEA. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------
    # Conditional DETR model and criterion classes.
    # Copyright (c) 2021 Microsoft. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------
    # Modified from DETR (https://github.com/facebookresearch/detr)
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
    # ------------------------------------------------------------------------
    # Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
    # Copyright (c) 2020 SenseTime. All Rights Reserved.
    # ------------------------------------------------------------------------
import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino_new.util import box_ops, get_tokenlizer
from groundingdino_new.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino_new.util.utils import get_phrases_from_posmap
from groundingdino_new.util.visualizer import COCOVisualizer
from groundingdino_new.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss
from maskrcnn_benchmark.structures.image_list import ImageList 
from maskrcnn_benchmark.modeling.rpn.inference import convert_grounding_to_od_logits
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_ml_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
# from groundingdino_new.util.inference import preprocess_caption
from maskrcnn_benchmark.modeling.poolers import CustomPooler, Pooler
from groundingdino_new.models.GroundingDINO.loss import SetCriterion
from groundingdino_new.models.GroundingDINO.matcher import build_matcher
from maskrcnn_benchmark.modeling.language_backbone import build_language_backbone
from maskrcnn_benchmark.modeling.language_backbone.modeling_bert_new import QVBertModel
from transformers import BertConfig, RobertaConfig, RobertaModel
from maskrcnn_benchmark.modeling.query_selector import build_query_selector

import os
from timm.models.layers import trunc_normal_

def expand_bbox(box_list, expand_ratio=1.5):
    new_box_list=[]
    for boxes in box_list:
        assert boxes.mode == "xyxy"
        bbox=boxes.bbox
        image_size=boxes.size
        box_w, box_h = bbox[:,2] - bbox[:,0], bbox[:,3] - bbox[:,1]
        new_box_w, new_box_h = box_w*expand_ratio, box_h*expand_ratio
        diff_w=(new_box_w-box_w)/2
        diff_h=(new_box_h-box_h)/2
        diff=torch.stack([-diff_w, -diff_h, diff_w, diff_h], dim=1)
        new_bbox=bbox+diff
        new_boxes=BoxList(new_bbox, image_size, mode="xyxy")
        labels=boxes.get_field('labels')
        new_boxes.add_field('labels', labels)
        new_boxes=new_boxes.clip_to_image(remove_empty=True)
        new_box_list.append(new_boxes)
    return new_box_list

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


# db
class CoOpModule(nn.Module):
    def __init__(self, prompt_length, prompt_channel, use_prompt=False, prompt=None) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_channel = prompt_channel
        if use_prompt:
            self.coop_prompt = prompt
        else:
            self.coop_prompt = nn.Parameter(torch.zeros(1, self.prompt_length, self.prompt_channel))
            trunc_normal_(self.coop_prompt, std=0.02)
    
    def forward(self, x):
        return x
    
class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        cfg = None,
        use_moe_lora = False, # db
        prompt_length=16, 
        prompt_channel=768,
        use_coop = False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.cfg = cfg
        self.box_threshold = cfg.GROUNDINGDINO.box_threshold
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present
        self.use_moe_lora = use_moe_lora

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # loss criterion
        self.loss_evaluator = SetCriterion(matcher=build_matcher(cfg.GROUNDINGDINO.matcher), cfg=cfg)


        # box pooler for extracting cache
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        if cfg.VISION_QUERY.SELECT_FPN_LEVEL:
            self.pooler = Pooler(
            output_size= (resolution, resolution) ,
            scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            use_v2=True,
            )
        else:
            self.pooler = CustomPooler(
                output_size= (resolution, resolution) ,
                scales=cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES,
                sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
                use_v2=True,
            )
        self.pool=nn.AvgPool2d(2)

        # query selector
        if cfg.VISION_QUERY.DISABLE_SELECTOR:
            self.query_selector = None
        else:
            self.query_selector = build_query_selector(cfg)

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        # if os.path.basename(text_encoder_type) != "bert-base-uncased":
        #     raise NotImplementedError
        # self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        config = BertConfig.from_pretrained(text_encoder_type)
        self.bert = QVBertModel.from_pretrained(text_encoder_type, dim_t=config.hidden_size, dim_v=self.hidden_dim, share_kv=cfg.VISION_QUERY.SHARE_KV, cfg=cfg, config=config)

        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze
        # db
        self.use_coop = use_coop
        if self.use_coop:
            self.prompt_length = prompt_length
            self.prompt_channel = prompt_channel
            self.coop = CoOpModule(prompt_length, prompt_channel)
        else:
            self.coop = None
        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    
    def convert_groundingdino_to_glip_output(self, groundingdino_out, positive_map, image_sizes):
        dot_product_logits = groundingdino_out['pred_logits']
        box_regression = groundingdino_out['pred_boxes']
        B, N, _ = dot_product_logits.shape
        box_cls = dot_product_logits.new_zeros(B, N, self.cfg.MODEL.DYHEAD.NUM_CLASSES - 1)
        # candidate_inds = dot_product_logits.max(dim=-1)[0] > self.box_threshold
        scores = convert_grounding_to_od_logits(logits=dot_product_logits, box_cls=box_cls,
                                                        positive_map=positive_map,
                                                        score_agg="MEAN",
                                                        )
        box_cls = scores

        candidate_inds = box_cls.max(dim=-1)[0] > self.box_threshold
        # pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        # pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        for per_box_cls, per_box_regression, per_candidate_inds, image_size \
                in zip(box_cls, box_regression, candidate_inds, image_sizes):
            per_box_cls = per_box_cls[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(1, sorted=False)

            per_class = top_k_indices[:, 0] + 1

            # print(per_class)

            box = per_box_regression[per_candidate_inds, :].view(-1, 4)
            H, W = image_size
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H]).to(box.device)[None, ...]
            # from xywh to xyxy
            box[:, :2] = box[:, :2] - box[:, 2:] / 2
            box[:, 2:] = box[:, 2:] + box[:, :2]

            detections = box

            boxlist = BoxList(detections, (W, H), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls[:,0])
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, min_size=0)
            results.append(boxlist)

        return results

    def load_query_bank(self, query_path):
        self.query_selector.load_query_bank(query_path)

    @torch.no_grad()
    def extract_query(self, 
        samples=None,
        targets=None,
        query_images=None, # default_dict(list) ,list[tensors] num_classes: (num_queries, num_scales, num_channels)
        visual_features=None,
        exclude_similar=False,
        device = None,
        max_query_number = None,
        ):
        device = device if device else samples.tensors.device
        targets = [target.to(device)
                    for target in targets if target is not None]
        # targets=expand_bbox(targets, expand_ratio=self.cfg.VISION_QUERY.EXPAND_RATIO)
        # ================= [ 修改开始 ] =================
        # 1. 备份 image_id
        # 因为 expand_bbox 可能会返回全新的 BoxList 对象从而丢失自定义字段
        img_ids_backup = []
        for t in targets:
            if t.has_field("image_id"):
                img_ids_backup.append(t.get_field("image_id"))
            else:
                img_ids_backup.append(None)

        # 2. 执行 expand_bbox (原操作)
        targets = expand_bbox(targets, expand_ratio=self.cfg.VISION_QUERY.EXPAND_RATIO)

        # 3. 恢复 image_id
        # 前提：expand_bbox 不会改变 targets 列表的长度和顺序
        for t, img_id in zip(targets, img_ids_backup):
            if img_id is not None:
                t.add_field("image_id", img_id)
        # ================= [ 修改结束 ] =================
        if visual_features is None:
            if isinstance(samples, ImageList):
                image_sizes = samples.image_sizes
                samples = samples.tensors
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples, image_sizes=image_sizes)
            features, poss = self.backbone(samples)

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                srcs.append(self.input_proj[l](src))
                masks.append(mask)
                assert mask is not None 
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    poss.append(pos_l)
            
            visual_features = srcs
        else:
            visual_features = [v.to(device) for v in visual_features]

        if self.cfg.VISION_QUERY.SELECT_FPN_LEVEL:
            query_feats=self.pooler(visual_features, targets) # num_boxes, num_channels, pooler_size, pooler_size
            query_feats=query_feats[None, ...] # 1, num_boxes, num_channels, pooler_size, pooler_size
        else:
            query_feats=self.pooler(visual_features, targets) # num_scales, num_boxes, num_channels, pooler_size, pooler_size
        
        # average different fpn levels
        if not self.cfg.VISION_QUERY.SELECT_FPN_LEVEL:
            assert len(visual_features) == len(query_feats) == 5 # TODO: support flexible level numbers
        query_feats = query_feats.mean(dim=[-2,-1]).permute(1, 0, 2) # num_boxes, num_scales, num_channels

        labels=torch.cat([t.get_field('labels') for t in targets])
        # db
        # [NEW] 获取对应的 image_id
        img_ids = []
        for t in targets:
            # 尝试获取 image_id，如果没有则设为 -1
            curr_id = t.get_field('image_id') if t.has_field('image_id') else -1
            img_ids.extend([curr_id] * len(t))
        assert len(labels)==len(query_feats)

        max_query_number = self.cfg.VISION_QUERY.MAX_QUERY_NUMBER if max_query_number is None else max_query_number
        # for label, feat in zip(labels, query_feats):
        #     label=label.item()
        #     num_queries=len(query_images[label])
        # db
        # [MODIFIED] 循环同时处理 label, feat 和 img_id
        for label, feat, img_id in zip(labels, query_feats, img_ids):
            label=label.item()
            
            # [修正点] 初始化结构为字典，并确保 features 维度正确 (0, num_scales, num_channels)
            if label not in query_images:
                # 使用 *feat.shape 自动解包 (num_scales, num_channels)
                query_images[label] = {
                    'features': torch.empty(0, *feat.shape, device=device), 
                    'ids': []
                }

            # 检查当前数量 (基于 features 的长度)
            # 注意：这里需要根据新的字典结构来获取长度
            # 为了兼容旧代码逻辑，如果 query_images[label] 还是 list/tensor (这不应该发生在新逻辑中)，这里会报错，确保清空了旧的 bank
            num_queries = len(query_images[label]['ids'])
            if num_queries >= max_query_number:
                continue
            if exclude_similar and num_queries > 0:
                assert feat.shape[0] == 1 # TODO: enable all-level and spacial features
                bank_features = F.normalize(query_images[label]['features'], p=2, dim=-1)  # db N, 1, C
                new_features = F.normalize(feat, p=2, dim=-1) # 1, C
                similarity = einsum('b n d, n d -> b n', bank_features, new_features)
                has_similar_in_bank = (similarity > self.cfg.VISION_QUERY.SIMILARITY_THRESHOLD).sum() > 0
                if has_similar_in_bank:
                    continue

            # if num_queries==0:
            #     query_images[label] = feat[None, ...]
            # else:
            #     query_images[label] = torch.cat([query_images[label], feat[None, ...]])
            # db
            query_images[label]['features'] = torch.cat([query_images[label]['features'], feat[None, ...]])
            query_images[label]['ids'].append(img_id)
        return query_images

    def flatten_fpn_features(self, features):
        # downsample and flat fpn features for pre-select in language backbone
        return torch.cat([self.pool(f).flatten(-2,-1) for i, f in enumerate(features)], dim=2).permute(0,2,1)

    @torch.no_grad()
    def get_labels_and_maps_from_positive_map(self, positive_map, dtype=torch.float):
        if isinstance(positive_map, torch.Tensor):
            positive_map_dict = {}
            for i in range(len(positive_map)):
                positive_map_dict[i + 1] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
            positive_map = positive_map_dict

        # Only for inference
        labels_in_caption=[k for k,v in positive_map.items() if len(v) !=0]
        num_labels=len(labels_in_caption)
        all_map = torch.zeros((num_labels, self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN), dtype=dtype, device=self.cfg.MODEL.DEVICE)
        for j, label in enumerate(labels_in_caption):
            position=positive_map[label]
            all_map[j, position] = 1 # inplace
        all_map = all_map / (all_map.sum(-1)[:, None] + 1e-6)
        return labels_in_caption, all_map
    
    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, ImageList):
            image_sizes = samples.image_sizes
            samples = samples.tensors
        # db 2. Extract Features Flag (For MR-GDINO Routing)
        # 优先检查参数
        extract_feat = kw.get('extract_feat', False)

        # 3. Text & Target Processing
        # 只有在非 extract_feat 模式下，才强制需要 captions 和 targets
        captions = None
        positive_map = None
        
        if not extract_feat:
            if targets is None:
                captions = kw.get("captions", None)
            else:
                captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
            
            if captions is not None:
                captions = [preprocess_caption(c) for c in captions]
            
            positive_map = kw.get('positive_map', None)
        # if targets is None:
        #     captions = kw["captions"]
        # else:
        #     captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        # len(captions)

        # captions = [preprocess_caption(c) for c in captions]


        # positive_map = kw['positive_map']
        try:
            return_backbone_features = kw['return_backbone_features']
        except:
            return_backbone_features = False

        # import ipdb; ipdb.set_trace()

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples, image_sizes=image_sizes)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # db# ================================================================
        # [NEW CODE START] MQ-My Routing Logic (LoRA Routing)
        # ================================================================
        # 只有当传入了任务库 (subtasks_mean_feat) 时才启用路由
        if "subtasks_mean_feat" in kw and "subtasks_lora_data" in kw:
            subtasks_mean_feat = kw["subtasks_mean_feat"]
            # subtasks_prompts_list = kw["subtasks_prompts_list"]
            # subtasks_lora_data = kw["subtasks_lora_data"]
            tau = kw.get('tau', 0.0) # 相似度阈值

            # 1. 提取当前图像的全局特征 (使用最高层级特征 features[-1])
            # features[-1].tensors shape: [B, C, H, W] -> Global Avg -> [B, C]
            # 注意：这里假设 batch_size=1 (推理时的常见设置)
            mean_feature = torch.mean(features[-1].tensors, dim=(2,3)).cpu()

            max_key = -1
            max_sim = -10000.0

            # 2. 路由计算：寻找最相似的任务
            for key in subtasks_mean_feat:
                # subtasks_mean_feat[key] 应该是 [C] 或 [1, C] 的 Tensor
                # F.cosine_similarity 会自动处理维度
                sim = F.cosine_similarity(subtasks_mean_feat[key].view(1, -1), mean_feature.view(1, -1))
                
                if sim > max_sim:
                    max_sim = sim
                    max_key = key
            
            # print(f"Routing Result: Selected Task [{max_key}] with Score {max_sim.item():.4f}")

            # 3. 阈值判断与参数加载
            if max_sim < tau:
                # 相似度过低，不使用特定任务的 LoRA (退回基础模型)
                # if hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'open_lora'):
                self.transformer.encoder.open_lora = False
                self.coop = None
            else:
                # 命中任务，加载对应的 LoRA 参数
                # 确保 Transformer 开启了 LoRA 开关
                # if hasattr(self.transformer, 'encoder') and hasattr(self.transformer.encoder, 'open_lora'):
                self.transformer.encoder.open_lora = True
                # if self.use_coop:
                #     if self.coop == None:
                #         self.coop = CoOpModule(self.prompt_length, self.prompt_channel).to(samples.device)
                #     select_prompt = subtasks_prompts_list[max_key]
                #     log = self.coop.load_state_dict(select_prompt, strict=False)
                #     # print(log)
                
                # if max_key in subtasks_lora_data:
                #     select_lora = subtasks_lora_data[max_key]
                #     # [关键] 动态热加载 LoRA 参数
                #     # strict=False 是必须的，因为 select_lora 只包含部分权重
                #     self.transformer.load_state_dict(select_lora, strict=False)
        # ================================================================
        # [NEW CODE END]
        # ================================================================
        # db
        # ================================================================
        # [NEW] MR-GDINO Routing Logic
        # 如果只需要提取图像全局特征用于持续学习的路由（Memory Retrieval）
        # 则在此处提前返回，跳过后续所有 Transformer/Text 逻辑
        if extract_feat:
            # 取最高语义层级的特征 (Backbone/FPN 的最后一层)
            # srcs: List[Tensor], Tensor shape [B, C, H, W]
            last_feat_map = srcs[-1]
            
            # Global Average Pooling -> [B, C]
            # 计算 H, W 维度的均值
            avg_feat = last_feat_map.mean(dim=[-2, -1])
            
            # 返回 CPU Tensor 以方便 pickle 存储
            return avg_feat.cpu()
        # ================================================================

        if self.cfg.VISION_QUERY.ENABLED:
            if self.training:
                batched_labels_in_caption=[t.get_field('labels_in_caption') for t in targets]
                batched_all_map=[t.get_field('all_map') for t in targets]
                batched_pos_category_map=[t.get_field('positive_category_map') for t in targets]
                ################ BUG: batched_pos_category_map is not binary ######################
                batched_pos_labels = [t.get_field('labels') for t in targets]
                batched_img_ids = [t.get_field('image_id') if t.has_field('image_id') else -1 for t in targets] # db
            else:
                assert samples.tensors.shape[0]==1 # TODO: Only support batch size = 1 for test
                labels_in_caption, all_map = self.get_labels_and_maps_from_positive_map(positive_map, dtype=srcs[0].dtype)
                batched_labels_in_caption = [labels_in_caption]
                batched_all_map = [all_map]
                batched_pos_category_map = None
                batched_pos_labels = None
                # [NEW] 测试时设为 None
                batched_img_ids = None
            # [MODIFIED] 将 batched_img_ids 传递给 query_selector
            query_features, query_attetion_masks, batched_has_vision_query = self.query_selector(
                batched_labels_in_caption, 
                batched_all_map, 
                batched_pos_labels,
                batched_img_ids=batched_img_ids # 新增参数
            )

            # query_features, query_attetion_masks, batched_has_vision_query=self.query_selector(batched_labels_in_caption, batched_all_map, batched_pos_labels)
 
            vision_inputs_in_language_backbone={'vision': query_features, 'images': self.flatten_fpn_features(srcs), 'vision_attention_mask': query_attetion_masks, 'batched_pos_category_map': batched_pos_category_map}
        else:
            vision_inputs_in_language_backbone={'vision': None, 'images': None, 'vision_attention_mask': None, 'batched_pos_category_map': None}


       # encoder texts
        # assume each category is consist of its text tokens and one '.'
        # tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
        #     samples.device
        # )
        tokenized = self.tokenizer(captions, padding='longest', return_tensors="pt").to(
            samples.device
        )
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer, self.coop
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        tokenized_for_encoder.update(vision_inputs_in_language_backbone)
        tokenized_for_encoder["coop"] = self.coop
        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        if self.coop: # 这里为什么coop是运行的
                encoded_text = torch.cat((encoded_text[:,:1], encoded_text[:,17:]), dim=1)

        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask
        if self.coop:
                position_ids = torch.cat((position_ids[:,:1], position_ids[:,17:]), dim=1)
                
                text_self_attention_masks = torch.cat((text_self_attention_masks[:,:1], text_self_attention_masks[:,17:]),dim=1)
                
                text_self_attention_masks = torch.cat((text_self_attention_masks[:,:,:1], text_self_attention_masks[:,:,17:]),dim=2)

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }





        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict, use_moe_lora=self.use_moe_lora # db
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        if self.training:
            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
            aux_outputs = [{"pred_logits": outputs_class[k], "pred_boxes": outputs_coord_list[k]} for k in range(len(outputs_class)-1)]
            out['aux_outputs'] = aux_outputs
            positive_map_ = positive_map.clone().to(outputs_class[-1].device)
            positive_map_[positive_map_>0]=1.

            # padding to max_text_len
            text_mask = torch.full((*text_dict["text_token_mask"].shape[:-1], self.max_text_len), bool(False), device=text_dict["text_token_mask"].device)
            text_mask[..., : text_dict["text_token_mask"].shape[-1]] = text_dict["text_token_mask"]

            losses = self.loss_evaluator(out, targets, text_mask=text_mask ,positive_map=positive_map_)

            if self.cfg.VISION_QUERY.ENABLED:
                #### gate loss #####
                # concatenate all gates
                gates = []
                for _ ,g in bert_output['vision_query_gates'].items():
                    gates = gates + g

                num_gates=len(gates)
                loss_gate=0
                for g in gates:
                    loss_gate=loss_gate+(1-torch.abs(g[0]))
                loss_gate= self.cfg.VISION_QUERY.GATE_REGULARIZATION_SCALE * loss_gate / num_gates
                if self.cfg.VISION_QUERY.GATE_REGULARIZATION:
                    gate_losses = {'loss_gate': loss_gate.sum()}
                else:
                    loss_gate = loss_gate.sum().detach() # Only for analysis
                    gate_losses = {'loss_gate': loss_gate}
                ####################

                losses.update(gate_losses)
            return losses
        else:
            out = {"pred_logits": outputs_class[-1].sigmoid(), "pred_boxes": outputs_coord_list[-1]}
            result = self.convert_groundingdino_to_glip_output(out, positive_map, image_sizes)
            if return_backbone_features:
                return result, srcs
            return result


        # # for intermediate outputs
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args, cfg):
    # ================= [核心修复] =================
    # 1. 临时解冻
    args.defrost() 
    
    # 2. 修改/注入参数
    # 如果你在 defaults.py 里注册了，这里就是覆盖值
    # 如果没注册，这里就是新增键
    # 强制使用 cfg 中的全局配置，或者直接写 True
    if hasattr(cfg.GROUNDINGDINO, "use_moe_lora"):
         args.use_moe_lora = cfg.GROUNDINGDINO.use_moe_lora
    else:
         args.use_moe_lora = True # 强制开启

    # 3. 重新冻结 (保持良好的编程习惯)
    args.freeze()
    # ============================================

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        cfg=cfg,
        use_moe_lora=args.use_moe_lora,
        use_coop=args.use_coop,
    )

    return model
