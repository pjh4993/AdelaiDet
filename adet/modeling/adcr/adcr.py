import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.utils.events import get_event_storage

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .adcr_outputs import ADCROutputs
import logging


__all__ = ["ADCR"]

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
class ADCR(nn.Module):
    """
    Implement ADCR (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.ADCR.IN_FEATURES
        self.fpn_strides = cfg.MODEL.ADCR.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.ADCR.YIELD_PROPOSAL

        self.adcr_head = ADCRHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.adcr_head.in_channels_to_top_module

        self.adcr_outputs = ADCROutputs(cfg)
        self.cnt = 0

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.adcr_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

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
        logits_pred, reg_pred, cid_pred, rid_pred, iou_pred, top_feats, bbox_towers = self.adcr_head(
            features, top_module, self.yield_proposal
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)
            }

        if self.training:
            results, losses = self.adcr_outputs.losses(
                logits_pred, reg_pred, cid_pred, rid_pred, iou_pred,
                locations, gt_instances, self.adcr_head.relation_net, top_feats
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.adcr_outputs.predict_proposals(
                        logits_pred, reg_pred, cid_pred, rid_pred, iou_pred,
                        locations, images.image_sizes, self.adcr_head.relation_net, top_feats
                    )


            PLCS_thr = self.adcr_outputs.PCLS_thr / (results['num_objects'] + 1e-6)
            PIoU_thr = self.adcr_outputs.PIoU_thr / (results['num_objects'] + 1e-6)
            CPSR = self.adcr_outputs.CPSR / (results['num_objects'] + 1e-6)
            RPSR = self.adcr_outputs.RPSR / (results['num_objects'] + 1e-6)
            CPMAX = self.adcr_outputs.CPMAX / (results['num_objects'] + 1e-6)
            RPMAX = self.adcr_outputs.RPMAX / (results['num_objects'] + 1e-6)

            get_event_storage().put_scalar("PLCS_thr", PLCS_thr)
            get_event_storage().put_scalar("PIoU_thr", PIoU_thr)
            get_event_storage().put_scalar("CPSR", CPSR)
            get_event_storage().put_scalar("RPSR", RPSR)
            get_event_storage().put_scalar("EMB_acc", self.adcr_outputs.EMB_acc)
            get_event_storage().put_scalar("PIOU_acc", sum(list(self.adcr_outputs.PIOU_acc.values())))
            get_event_storage().put_scalar("psr_rate", self.adcr_outputs.positive_sample_rate)

            self.cnt+=1
            if self.cnt % 20 == 0:
                logging.getLogger(__name__).info(
                    'CLS_thr: {:4f} IoU_tr: {:4f} CPSR: {:4f} RPSR: {:4f} EMB_acc: {:4f} CPMAX: {:4f} RPMAX {:4f} pss_rate: {:4f}'.format(
                        PLCS_thr, PIoU_thr, CPSR, RPSR,
                        self.adcr_outputs.EMB_acc,
                        CPMAX, RPMAX,
                        self.adcr_outputs.positive_sample_rate
                    )
                )
                logging.getLogger(__name__).info(
                    "PIOU_acc: " + str(self.adcr_outputs.PIOU_acc)
                )

            self._detect_anomaly(sum(list(losses.values())), losses)

            return results, losses
        else:
            results = self.adcr_outputs.predict_proposals(
                logits_pred, reg_pred, cid_pred, rid_pred, iou_pred,
                locations, images.image_sizes, self.adcr_head.relation_net, top_feats
            )

            return results, {}

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN loss_dict = {}".format(
                    loss_dict
                )
            )

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


class ADCRHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.ADCR.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.ADCR.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.ADCR.NUM_CLS_CONVS,
                                cfg.MODEL.ADCR.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.ADCR.NUM_BOX_CONVS,
                                 cfg.MODEL.ADCR.USE_DEFORMABLE),
                        "emb": (cfg.MODEL.ADCR.NUM_EMB_CONVS,
                                  cfg.MODEL.ADCR.USE_DEFORMABLE)
                        }
        norm = None if cfg.MODEL.ADCR.NORM == "none" else cfg.MODEL.ADCR.NORM
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
        self.cid_emb = nn.Conv2d(
            in_channels, cfg.MODEL.ADCR.EMB_DIM , kernel_size=3,
            stride=1, padding=1
        )
        self.rid_emb = nn.Conv2d(
            in_channels, cfg.MODEL.ADCR.EMB_DIM , kernel_size=3,
            stride=1, padding=1
        )
        self.iou_pred = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )
        self.relation_net = nn.Conv1d(
            cfg.MODEL.ADCR.EMB_DIM * 2, 1, kernel_size=1
        )

        if cfg.MODEL.ADCR.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower, self.emb_tower,
            self.cls_logits, self.bbox_pred, self.iou_pred, self.relation_net
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        for modules in [
            self.cid_emb, self.rid_emb,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.1)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ADCR.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.relation_net.bias, bias_value)
        torch.nn.init.constant_(self.iou_pred.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        iou_pred = []
        cid_pred = []
        rid_pred = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            emb_tower = self.emb_tower(feature)

            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            cid_pred.append(self.cid_emb(emb_tower))
            rid_pred.append(self.rid_emb(emb_tower))
            iou_pred.append(self.iou_pred(bbox_tower))

            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved ADCR, instead of exp.
            bbox_reg.append(F.relu(reg))

            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, cid_pred, rid_pred, iou_pred, top_feats, bbox_towers
