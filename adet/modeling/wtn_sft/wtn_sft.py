import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .wtn_sft_outputs import WTN_SFTOutputs


__all__ = ["WTN_SFT"]

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
class WTN_SFT(nn.Module):
    """
    Implement WTN_SFT (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.WTN_SFT.IN_FEATURES
        self.fpn_strides = cfg.MODEL.WTN_SFT.FPN_STRIDES

        self.wtn_sft_head = WTN_SFTHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.wtn_sft_head.in_channels_to_top_module

        self.wtn_sft_outputs = WTN_SFTOutputs(cfg)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness = self.wtn_sft_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred  = self.wtn_sft_head(features)

        results = {}

        if self.training:
            results, losses = self.wtn_sft_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances
            )
            
            return results, losses
        else:
            results = self.wtn_sft_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes
            )

            return results, {}

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


class WTN_SFTHead(nn.Module):
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

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
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
            self.cls_tower, self.bbox_tower, self.share_tower, self.sft_tower,
            self.cls_logits, self.bbox_pred, self.ctrness, 
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.WTN_SFT.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        # initialize the bias for centerness
        torch.nn.init.constant_(self.ctrness.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)

            bbox_tower = self.bbox_tower(feature)

            sft_bbox_tower = self.sft_tower(cls_tower + bbox_tower) + bbox_tower

            logits.append(self.cls_logits(cls_tower))

            if self.ctrness_on_bbox:
                ctrness.append(self.ctrness(sft_bbox_tower))
            else:
                ctrness.append(self.ctrness(cls_tower))

            reg = self.bbox_pred(sft_bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved WTN_SFT, instead of exp.
            bbox_reg.append(F.relu(reg))
        
        return logits, bbox_reg, ctrness