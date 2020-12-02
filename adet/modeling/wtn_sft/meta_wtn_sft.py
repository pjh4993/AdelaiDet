import math
import numpy as np
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm, cat
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.utils.comm import get_world_size
from adet.utils.comm import reduce_sum

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .meta_wtn_sft_outputs import META_WTN_SFTOutputs, compute_ctrness_targets
from detectron2.structures.image_list import ImageList

__all__ = ["META_WTN_SFT"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class META_WTN_SFT(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.WTN_SFT.IN_FEATURES
        self.fpn_strides = cfg.MODEL.WTN_SFT.FPN_STRIDES

        self.feature_extractor = META_WTN_SFT_Head(cfg, [input_shape[f] for f in self.in_features])

        self.meta_wtn_sft_outputs = META_WTN_SFTOutputs(cfg)

        self.n_way = cfg.DATASAMPLER.CLASSWISE_SAMPLER.N_WAY
        self.k_shot = cfg.DATASAMPLER.CLASSWISE_SAMPLER.K_SHOT
        self.q_query = cfg.DATASAMPLER.CLASSWISE_SAMPLER.Q_QUERY

    def forward(self, batched_images, batched_features, batched_gt_instances):
        results = []
        losses = {
            "loss_wtn_sft_cls": [],
            "loss_wtn_sft_loc": [],
            "loss_wtn_sft_ctr": [],
        }

        for images, features, gt_instances in zip(batched_images, batched_features, batched_gt_instances):

            features = [features[f] for f in self.in_features]

            locations = self.compute_locations(features)

            cls_features, bbox_features = self.feature_extractor.forward_feature(features)

            supp_set, query_set = self.split_by_sampler(images, cls_features, bbox_features, gt_instances)

            prototypes = self.calculate_prototype(supp_set, locations, gt_instances['labels'])

            pred_logits, pred_deltas, pred_ctrness = self.feature_extractor.forward_with_prototype(query_set, prototypes)

            if self.training:
                extra, loss = self.meta_wtn_sft_outputs.losses(pred_logits, pred_deltas, pred_ctrness, locations, query_set["gt_instances"], gt_instances["labels"])
                for k, v in loss.items():
                    losses[k].append(loss[k])
            else:
                pass

        for k, v in losses.items():
            losses[k] = torch.stack(v).mean()

        return results, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def calculate_prototype(self, supp_set, locations, gt_labels):
        instances = self.meta_wtn_sft_outputs.get_prototype_target(locations, supp_set['gt_instances'], supp_set, gt_labels)
        labels = instances.labels.flatten()
        cls_features = instances.cls_features

        pos_per_label = {k.item() : torch.nonzero(labels == k).squeeze(1) for k in gt_labels}
        ctrness_targets = compute_ctrness_targets(instances.reg_targets)
        class_prototypes = {}

        for k, pos in pos_per_label.items():
            assert len(pos) != 0
            k_cls_feature = cls_features[pos]
            k_ctrness = ctrness_targets[pos].reshape(-1,1)
            k_ctrness[k_ctrness.isnan()] = 0
            k_ctrness = nn.Softmax(dim=0)(k_ctrness)
            prototype = (k_cls_feature * k_ctrness).sum(dim=0)

            assert prototype.isnan().sum() == 0

            class_prototypes[k] = prototype

        return class_prototypes

    def split_by_sampler(self, images, cls_features, bbox_features, gt_instances):
        n_per = self.k_shot + self.q_query
        supp_set = {
            "cls_features": [],
            "bbox_features": [],
            "images" : [],
            "gt_instances": [],
        }
        query_set = {
            "cls_features": [],
            "bbox_features": [],
            "images" : [],
            "gt_instances": [],
        }

        supp_set_idx = []
        query_set_idx = []

        for _id in range(self.n_way):
            id_list = np.arange(_id * n_per, (_id +1) * n_per)
            supp_set_idx.extend(id_list[:self.k_shot])
            query_set_idx.extend(id_list[self.k_shot:])

        supp_set['cls_features'] = [x[supp_set_idx] for x in cls_features]
        supp_set['bbox_features'] = [x[supp_set_idx] for x in bbox_features]
        supp_set['images'] = images[supp_set_idx]
        supp_set['gt_instances'] = [gt_instances["instances"][x] for x in supp_set_idx]

        query_set['cls_features'] = [x[query_set_idx] for x in cls_features]
        query_set['bbox_features'] = [x[query_set_idx] for x in bbox_features]
        query_set['images'] = images[query_set_idx]
        query_set['gt_instances'] = [gt_instances["instances"][x] for x in query_set_idx]

        return supp_set, query_set

class META_WTN_SFT_Head(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.WTN_SFT.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.WTN_SFT.FPN_STRIDES
        self.ctrness_on_bbox = cfg.MODEL.WTN_SFT.CTRNESS_ON_BBOX
        head_configs = {"cls": (cfg.MODEL.WTN_SFT.NUM_CLS_CONVS,
                                cfg.MODEL.WTN_SFT.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.WTN_SFT.NUM_BOX_CONVS,
                                 cfg.MODEL.WTN_SFT.USE_DEFORMABLE),
                        "share": (cfg.MODEL.WTN_SFT.NUM_SHARE_CONVS,
                                  False),
                        "sft": (cfg.MODEL.WTN_SFT.NUM_SFT_CONVS, False),
                        "relation": (cfg.MODEL.WTN_SFT.NUM_SFT_CONVS, False),
                        }
        norm = None if cfg.MODEL.WTN_SFT.NORM == "none" else cfg.MODEL.WTN_SFT.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.WTN_SFT.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower, self.share_tower, self.sft_tower, self.relation_tower,
            self.bbox_pred, self.ctrness,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.WTN_SFT.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # initialize the bias for centerness
        torch.nn.init.constant_(self.ctrness.bias, bias_value)

    def forward_feature(self, x):
        cls_features = []
        bbox_features = []

        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)

            bbox_tower = self.bbox_tower(feature)

            cls_features.append(cls_tower)
            bbox_features.append(bbox_tower)

        return cls_features, bbox_features

    def forward_with_prototype(self, query_set, cls_prototypes):

        cls_features = query_set["cls_features"]
        box_features = query_set["bbox_features"]

        logits = []
        bbox_reg = []
        ctrness = []
        for l, (cls_tower, bbox_tower) in enumerate(zip(cls_features, box_features)):
            dist_per_cls = []
            for k, v in cls_prototypes.items():
                v = v.reshape(1, cls_tower.shape[1], 1, 1).expand_as(cls_tower)
                dist_per_cls.append(self.relation_tower(cls_tower + v))

            dist = cat(dist_per_cls, dim=1)
            print(dist.min(), dist.max())
            assert dist.isnan().sum() == 0

            logits.append(-dist)

            sft_bbox_tower = self.sft_tower(cls_tower + bbox_tower) + bbox_tower

            if self.ctrness_on_bbox:
                ctrness.append(self.ctrness(sft_bbox_tower))
            else:
                ctrness.append(self.ctrness(cls_tower))

            reg = self.bbox_pred(sft_bbox_tower)

            if self.scales is not None:
                reg = self.scales[l](reg)

            assert reg.isnan().sum() == 0
            # Note that we use relu, as in the improved WTN_SFT, instead of exp.
            bbox_reg.append(F.relu(reg))


        return logits, bbox_reg, ctrness
