# import torch
# from torch import nn
# import numpy as np
# from torch.nn.utils.rnn import pad_sequence
# import random
# import os

# class QuerySelector(nn.Module):
#     def __init__(self, cfg):
#         super(QuerySelector, self).__init__()
#         self.device = torch.device(cfg.MODEL.DEVICE)
#         if not os.path.exists(cfg.VISION_QUERY.QUERY_BANK_PATH):
#             assert cfg.VISION_QUERY.QUERY_BANK_PATH == "", "query bank path {} not exists".format(cfg.VISION_QUERY.QUERY_BANK_PATH)
#             assert not cfg.VISION_QUERY.LEARNABLE_BANK
#             self.query_bank = None
#         else:
#             if cfg.VISION_QUERY.LEARNABLE_BANK:
#                 query_bank = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location='cpu')
#                 # add qv_layer to name only for easier parameter freezing
#                 self.query_bank = nn.ParameterDict({str(k): nn.Parameter(v) for k, v in query_bank.items()})
#                 query_dim = query_bank[list(query_bank.keys())[0]].shape[-1]
#             # else:
#             #     # add qv_layer to name only for easier parameter freezing
#             #     self.query_bank = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location=self.device) # default dict: num_classes [num_queries, num_scales, num_channels ]
#             #     query_dim = self.query_bank[list(self.query_bank.keys())[0]].shape[-1]
#             else:
#                 # ================= [核心修改开始] =================
#                 print(f"Loading Query Bank from {cfg.VISION_QUERY.QUERY_BANK_PATH} ...")
#                 # 1. 先加载原始数据
#                 raw_data = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location=self.device)
                
#                 # 2. 检查数据结构：是旧版(Tensor)还是新版(Dict with 'features')
#                 first_key = list(raw_data.keys())[0]
#                 first_value = raw_data[first_key]

#                 if isinstance(first_value, dict) and "features" in first_value:
#                     # ---> 命中新版结构
#                     print("Identified Query Bank with Image IDs. Enabling self-exclusion.")
#                     self.has_image_ids = True
                    
#                     # 拆分：Features 给 query_bank，IDs 给 query_ids
#                     self.query_bank = {k: v["features"] for k, v in raw_data.items()}
#                     self.query_ids = {k: v["ids"] for k, v in raw_data.items()}
                    
#                     # 现在 self.query_bank[k] 是 Tensor 了，可以取 shape
#                     query_dim = self.query_bank[first_key].shape[-1]
#                 else:
#                     # ---> 命中旧版结构 (兼容性)
#                     print("Identified Legacy Query Bank (No Image IDs).")
#                     self.has_image_ids = False
#                     self.query_bank = raw_data
#                     self.query_ids = None
#                     query_dim = self.query_bank[first_key].shape[-1]
#                 # ================= [核心修改结束] =================
#         if cfg.VISION_QUERY.ADD_VISION_LAYER:
#             self.tunable_vision_linear = torch.nn.Linear(query_dim, 1000, bias=False)
#             self.tunable_vision_linear.weight.data.fill_(0.0)
#         self.pure_text_rate = cfg.VISION_QUERY.PURE_TEXT_RATE
#         self.num_query_per_class = cfg.VISION_QUERY.NUM_QUERY_PER_CLASS
#         self.cfg = cfg
    
#     def load_query_bank(self, bank_path):
#         assert not self.cfg.VISION_QUERY.LEARNABLE_BANK
#         assert not self.cfg.VISION_QUERY.ADD_VISION_LAYER
#         # add qv_layer to name only for easier parameter freezing
#         self.query_bank = torch.load(bank_path, map_location=self.device) # default dict: num_classes [num_queries, num_scales, num_channels ]
    
#     # @torch.no_grad()
#     def forward(self, batched_label_list, batched_location_map, batched_pos_labels = None, batched_img_ids=None):
#         '''
#         Return query features, attention mask

#         batched_label_list: [[list]] - batch_size, num_labels
#         batched_location_map: [torch.tensor] one-hot -  batch_size, (num_labels, num_text_tokens)
#         '''
#         if self.query_bank is None:
#             return None, None, None

#         batched_queries = []
#         batched_queries_attn_mask = []
#         batched_has_vision_query = []
#         # db
#         # 如果没有传 img_ids (比如测试时)，用 None 填充
#         if batched_img_ids is None:
#             batched_img_ids = [None] * len(batched_label_list)

#         for k, (label_list, location_map, current_img_id) in enumerate(zip(batched_label_list, batched_location_map, batched_img_ids)):
#             query_per_image = []
#             mask_per_image = []
#             has_vision_query = []
#             for label, loc_map in zip(label_list, location_map):
#                 loc_map = loc_map.to(self.device)
#                 if self.cfg.VISION_QUERY.LEARNABLE_BANK:
#                     bank_item = self.query_bank[str(label)]
#                 else:
#                     bank_item = self.query_bank[label]
#                 # db
#                  # [NEW] 兼容性处理：检查 bank_item 是 Tensor 还是 Dict
#                 # 如果是旧版本生成的 bank (Tensor)，则没有 ids，无法过滤 (或者 ids 为空)
#                 if isinstance(bank_item, dict) and 'features' in bank_item:
#                     candidate_queries = bank_item['features']
#                     candidate_ids = bank_item['ids']
#                 else:
#                     candidate_queries = bank_item
#                     candidate_ids = None # 无法过滤

#                 num_total_queries=len(candidate_queries)
#                 loc_map = loc_map [None, ...] # 1, num_text_tokens

#                 # num_query_per_class = self.num_query_per_class
#                 num_query_per_class = np.random.choice(range(1, self.num_query_per_class+1)) if (self.cfg.VISION_QUERY.RANDOM_KSHOT and self.training) else self.num_query_per_class
#                 # num_queries = min(num_total_queries, num_query_per_class)
#                 # [KEY LOGIC] 过滤逻辑：找出不等于当前 image_id 的 query 索引
#                 # db
#                 valid_indices = list(range(num_total_queries))
#                 if self.training and candidate_ids is not None and current_img_id is not None:
#                     # 排除掉来源 ID 与当前训练图像 ID 相同的 query
#                     valid_indices = [i for i in valid_indices if candidate_ids[i] != current_img_id]
                
#                 # 如果过滤后没有可用的 query (极端情况，比如只有 1-shot 且就是当前图)，则回退或取空
#                 if len(valid_indices) == 0:
#                      # 这种情况下无法提供 visual query，只能使用纯文本
#                      num_queries = 0
#                 else:
#                      num_queries = min(len(valid_indices), num_query_per_class)
#                 if (random.random() < self.pure_text_rate) and self.training:
#                     # data augmentation: random select some labels for only text inputs, without vision query
#                     num_queries = 0

#                 # idx= np.random.choice(num_total_queries, num_queries, replace=False).tolist()
#                 # 从有效的 indices 中采样
#                 if num_queries > 0:
#                     selected_indices_idx = np.random.choice(len(valid_indices), num_queries, replace=False)
#                     idx = [valid_indices[i] for i in selected_indices_idx]
#                 else:
#                     idx = []

#                 if not self.training:
#                     idx = sorted(idx)
#                 if isinstance(candidate_queries, list):
#                     assert len(idx) == 0
#                 else:
#                     queries = candidate_queries[idx]
#                     num_scale=queries.shape[1]
#                     queries=queries.flatten(0,1)
#                     queries_attn_mask = loc_map.expand(num_queries*num_scale, -1)
#                     query_per_image.append(queries)
#                     mask_per_image.append(queries_attn_mask)

#                 if batched_pos_labels is None:
#                     pos_flag = True
#                 else:
#                     pos_flag = (label in batched_pos_labels[k])

#                 if pos_flag:
#                     has_vision_query.append(1 if num_queries > 0 else 0)

#             query_per_image=torch.cat(query_per_image)
#             mask_per_image=torch.cat(mask_per_image)
            
#             if self.cfg.VISION_QUERY.ADD_VISION_LAYER:
#                 query_per_image = self.tunable_vision_linear.weight[:query_per_image.size(0), :] + query_per_image


#             batched_queries.append(query_per_image)
#             batched_queries_attn_mask.append(mask_per_image)
#             batched_has_vision_query.append(has_vision_query)

        
#         batched_queries=pad_sequence(batched_queries, batch_first=True) # TODO: more efficiet implement
#         batched_queries_attn_mask=pad_sequence(batched_queries_attn_mask, batch_first=True)
        
#         # The batched_location_map averages the scores, for example, 'apple pie' has two tokenized tokens, thus the location map is (0.5, 0.5) rather than (1, 1). 
#         # So we reformulate the batched_queries_attn_mask to 0 or 1.
#         batched_queries_attn_mask[batched_queries_attn_mask!=0] = 1



#         return batched_queries, batched_queries_attn_mask, batched_has_vision_query

        
# db
import torch
from torch import nn
import os
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence

class QuerySelector(nn.Module):
    def __init__(self, cfg):
        super(QuerySelector, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.has_image_ids = False # 初始化标志位
        self.query_ids = None      # 初始化 ID 容器

        if not os.path.exists(cfg.VISION_QUERY.QUERY_BANK_PATH):
            assert cfg.VISION_QUERY.QUERY_BANK_PATH == "", "query bank path {} not exists".format(cfg.VISION_QUERY.QUERY_BANK_PATH)
            assert not cfg.VISION_QUERY.LEARNABLE_BANK
            self.query_bank = None
        else:
            if cfg.VISION_QUERY.LEARNABLE_BANK:
                # Learnable Bank 逻辑保持不变 (通常用于预训练，暂不涉及 ID 过滤)
                query_bank = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location='cpu')
                self.query_bank = nn.ParameterDict({str(k): nn.Parameter(v) for k, v in query_bank.items()})
                query_dim = query_bank[list(query_bank.keys())[0]].shape[-1]
            else:
                # ================= [核心修改：初始化加载逻辑] =================
                print(f"Loading Query Bank from {cfg.VISION_QUERY.QUERY_BANK_PATH} ...")
                # 1. 加载原始数据
                raw_data = torch.load(cfg.VISION_QUERY.QUERY_BANK_PATH, map_location=self.device)
                
                # 2. 检查数据结构
                # 为了安全，先获取第一个 Key
                first_key = list(raw_data.keys())[0]
                first_value = raw_data[first_key]

                # 判断是否包含 features 和 ids
                if isinstance(first_value, dict) and "features" in first_value:
                    print("✅ Identified Query Bank with Image IDs. Enabling self-exclusion.")
                    self.has_image_ids = True
                    
                    # 拆分数据：
                    # self.query_bank 只存储特征 Tensor
                    self.query_bank = {k: v["features"] for k, v in raw_data.items()}
                    # self.query_ids 只存储对应的 ID (List or Tensor)
                    self.query_ids = {k: v["ids"] for k, v in raw_data.items()}
                    
                    query_dim = self.query_bank[first_key].shape[-1]
                else:
                    print("⚠️ Identified Legacy Query Bank (No Image IDs). Anti-leakage disabled.")
                    self.has_image_ids = False
                    self.query_bank = raw_data
                    self.query_ids = None
                    # 兼容旧版，直接取 shape
                    if isinstance(first_value, torch.Tensor):
                         query_dim = first_value.shape[-1]
                    else:
                         query_dim = first_value['features'].shape[-1] # 极端情况防守
                # ================= [修改结束] =================

        if cfg.VISION_QUERY.ADD_VISION_LAYER:
            self.tunable_vision_linear = torch.nn.Linear(query_dim, 1000, bias=False)
            self.tunable_vision_linear.weight.data.fill_(0.0)
        
        self.pure_text_rate = cfg.VISION_QUERY.PURE_TEXT_RATE
        self.num_query_per_class = cfg.VISION_QUERY.NUM_QUERY_PER_CLASS
        self.cfg = cfg
    
    def load_query_bank(self, bank_path):
        # 即使重载 Bank，也建议复用 __init__ 中的拆分逻辑，这里暂时简化
        assert not self.cfg.VISION_QUERY.LEARNABLE_BANK
        self.query_bank = torch.load(bank_path, map_location=self.device)

    def forward(self, batched_label_list, batched_location_map, batched_pos_labels = None, batched_img_ids=None):
        '''
        Return query features, attention mask
        '''
        if self.query_bank is None:
            return None, None, None

        batched_queries = []
        batched_queries_attn_mask = []
        batched_has_vision_query = []

        # 兼容性处理：如果没有传 img_ids，填充 None
        if batched_img_ids is None:
            batched_img_ids = [None] * len(batched_label_list)

        for k, (label_list, location_map, current_img_id) in enumerate(zip(batched_label_list, batched_location_map, batched_img_ids)):
            query_per_image = []
            mask_per_image = []
            has_vision_query = []
            
            # 当前图的正样本集合 (用于判断 has_vision_query)
            current_pos_labels_set = set(batched_pos_labels[k].tolist()) if batched_pos_labels is not None else None

            for label, loc_map in zip(label_list, location_map):
                loc_map = loc_map.to(self.device)
                
                # 1. 获取 Key
                label_key = str(label) if self.cfg.VISION_QUERY.LEARNABLE_BANK else label
                
                # # 2. 从 Bank 获取特征和 ID
                # # ================= [核心修改：Forward 获取逻辑] =================
                # if label_key not in self.query_bank:
                #     # 银行里没存这个类，跳过
                #     num_queries = 0
                #     candidate_queries = torch.empty(0, 0, 0) # Placeholder
                #     candidate_ids = None
                # else:
                #     # 直接获取特征 (在 __init__ 已经保证是 Tensor 了)
                #     candidate_queries = self.query_bank[label_key]
                    
                #     # 获取 IDs (如果有的话)
                #     if self.has_image_ids and self.query_ids is not None:
                #         candidate_ids = self.query_ids[label_key]
                #     else:
                #         candidate_ids = None
                # # ================= [修改结束] =================
                # db 为了解决验证报错
                # ================= [ 修改开始 ] =================
                if label_key not in self.query_bank:
                    # 银行里没存这个类，跳过
                    num_queries = 0
                    candidate_queries = torch.empty(0, 0, 0) 
                    candidate_ids = None
                else:
                    # 获取原始数据 (可能是 Tensor，也可能是 Dict)
                    bank_item = self.query_bank[label_key]
                    
                    # 🔎 强校验逻辑：确保 candidate_queries 最终是 Tensor
                    if isinstance(bank_item, dict) and 'features' in bank_item:
                        # 如果是字典，说明 __init__ 没拆分，或者数据本身就是字典
                        candidate_queries = bank_item['features'] # 提取 Tensor
                        candidate_ids = bank_item.get('ids', None) # 提取 IDs
                    elif isinstance(bank_item, torch.Tensor):
                        # 如果已经是 Tensor
                        candidate_queries = bank_item
                        # 尝试从外部 self.query_ids 获取 ID (如果 __init__ 拆分过)
                        if self.has_image_ids and self.query_ids is not None:
                             candidate_ids = self.query_ids.get(label_key, None)
                        else:
                             candidate_ids = None
                    else:
                        # 异常情况处理
                        print(f"⚠️ Warning: Unknown bank item type: {type(bank_item)}")
                        candidate_queries = bank_item
                        candidate_ids = None
                # ================= [ 修改结束 ] =================

                num_total_queries = len(candidate_queries)
                loc_map = loc_map[None, ...] # 1, num_text_tokens

                # 3. 确定 K-Shot 数量
                num_query_per_class = self.num_query_per_class
                if self.cfg.VISION_QUERY.RANDOM_KSHOT and self.training:
                    num_query_per_class = np.random.choice(range(1, self.num_query_per_class+1))

                # 4. [防泄露过滤逻辑] (Anti-Leakage)
                valid_indices = list(range(num_total_queries))
                
                # 只有当：正在训练 + 有ID数据 + 当前图片有ID 时，才过滤
                if self.training and candidate_ids is not None and current_img_id is not None:
                    # 安全转换当前图片 ID 为 python int (处理 Tensor)
                    curr_id_val = current_img_id.item() if hasattr(current_img_id, "item") else current_img_id
                    
                    # 重新构建有效索引列表
                    filtered_indices = []
                    for i, src_id in enumerate(candidate_ids):
                        # 安全转换来源 ID
                        src_id_val = src_id.item() if hasattr(src_id, "item") else src_id
                        
                        # [核心判断] 来源 ID 不等于 当前 ID，才是合法的参考图
                        if src_id_val != curr_id_val:
                            filtered_indices.append(i)
                    valid_indices = filtered_indices

                # 5. 采样逻辑
                if len(valid_indices) == 0:
                     num_queries = 0
                else:
                     num_queries = min(len(valid_indices), num_query_per_class)
                
                if (random.random() < self.pure_text_rate) and self.training:
                    num_queries = 0

                idx = []
                if num_queries > 0:
                    selected_indices_idx = np.random.choice(len(valid_indices), num_queries, replace=False)
                    idx = [valid_indices[i] for i in selected_indices_idx]
                else:
                    idx = []

                if not self.training:
                    idx = sorted(idx)

                # 6. 构建最终特征
                if num_queries > 0:
                    # 使用索引提取特征
                    queries = candidate_queries[idx] # [K, S, C]
                    
                    # Flatten scale dimension
                    num_scale = queries.shape[1]
                    queries = queries.flatten(0,1)
                    
                    # Expand mask
                    queries_attn_mask = loc_map.expand(num_queries*num_scale, -1)
                    
                    query_per_image.append(queries)
                    mask_per_image.append(queries_attn_mask)

                # 7. 正负样本标记
                # 判断当前 Label 是否在 GT 里
                if current_pos_labels_set is None:
                    pos_flag = True
                else:
                    pos_flag = label in current_pos_labels_set

                if pos_flag:
                    has_vision_query.append(1 if num_queries > 0 else 0)

            # 拼接单张图的所有 Query
            if len(query_per_image) > 0:
                query_per_image = torch.cat(query_per_image)
                mask_per_image = torch.cat(mask_per_image)
                
                if self.cfg.VISION_QUERY.ADD_VISION_LAYER:
                    query_per_image = self.tunable_vision_linear.weight[:query_per_image.size(0), :] + query_per_image
            else:
                # 处理空情况
                query_per_image = torch.empty(0, 256).to(self.device) # 假设 dim=256，具体需动态获取
                if len(batched_location_map) > 0:
                     mask_width = batched_location_map[0].shape[0] 
                else: 
                     mask_width = 256
                mask_per_image = torch.empty(0, mask_width).to(self.device)

            batched_queries.append(query_per_image)
            batched_queries_attn_mask.append(mask_per_image)
            batched_has_vision_query.append(has_vision_query)

        # Pad Batch
        batched_queries = pad_sequence(batched_queries, batch_first=True)
        batched_queries_attn_mask = pad_sequence(batched_queries_attn_mask, batch_first=True)
        
        # Binary Mask
        batched_queries_attn_mask[batched_queries_attn_mask!=0] = 1

        return batched_queries, batched_queries_attn_mask, batched_has_vision_query


        

