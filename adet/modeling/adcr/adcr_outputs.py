from enum import unique
import logging
import torch
from torch import nn
import torch.nn.functional as F
import copy
import os

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes, pairwise_giou
from detectron2.utils.comm import get_world_size
from detectron2.modeling import ROIPooler
from fvcore.nn import sigmoid_focal_loss_jit
import torch.multiprocessing as mp

from adet.utils.comm import reduce_sum
from adet.layers import ml_nms, IOULoss


logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];
    
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class ADCROutputs(nn.Module):
    def __init__(self, cfg):
        super(ADCROutputs, self).__init__()

        self.focal_loss_alpha = cfg.MODEL.ADCR.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.ADCR.LOSS_GAMMA
        self.center_sample = cfg.MODEL.ADCR.CENTER_SAMPLE
        self.radius = cfg.MODEL.ADCR.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.ADCR.INFERENCE_TH_TRAIN
        self.pre_nms_iou_thresh_train = cfg.MODEL.ADCR.IOU_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.ADCR.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.ADCR.POST_NMS_TOPK_TRAIN
        self.loc_loss_func = IOULoss(cfg.MODEL.ADCR.LOC_LOSS_TYPE)

        self.pre_nms_thresh_test = cfg.MODEL.ADCR.INFERENCE_TH_TEST
        self.pre_nms_iou_thresh_test = cfg.MODEL.ADCR.IOU_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.ADCR.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.ADCR.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.ADCR.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.ADCR.THRESH_WITH_CTR

        self.num_classes = cfg.MODEL.ADCR.NUM_CLASSES
        self.strides = cfg.MODEL.ADCR.FPN_STRIDES
        self.tss = cfg.MODEL.ADCR.TSS
        self.emb_dim = cfg.MODEL.ADCR.EMB_DIM

        self.positive_sample_rate = cfg.MODEL.ADCR.POS_SAMPLE_RATE
        self.pss_diff = (1 - self.positive_sample_rate) / (1.5 * cfg.SOLVER.MAX_ITER)
        self.in_cb, self.ext_cb = cfg.MODEL.ADCR.IN_CB, cfg.MODEL.ADCR.EXT_CB
        self.pooler = ROIPooler(output_size=1, scales=[1/x for x in self.strides], sampling_ratio=0, pooler_type='ROIAlignV2')
        self.focal_piou = cfg.MODEL.ADCR.FOCAL_PIOU

        self.RPSR = 0
        self.CPSR = 0
        self.PIoU_thr = 0
        self.PCLS_thr = 0

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.ADCR.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances, logits_pred, reg_pred):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        if self.tss == "ATSS":
            training_targets, num_objects = self.compute_targets_ATSS(
                locations, gt_instances, loc_to_size_range, num_loc_list
            )
        elif self.tss == "DATSS":
            training_targets, num_objects = self.compute_targets_ATSS(
                locations, gt_instances, loc_to_size_range, num_loc_list, logits_pred, reg_pred
            )
        else:
            training_targets = self.compute_targets_for_locations(
                locations, gt_instances, loc_to_size_range, num_loc_list,
            )


        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets, num_objects

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # lr_max = reg_targets_per_im[:,:,[0,2]].max(dim=2)[0]
            # tb_max = reg_targets_per_im[:,:,[1,3]].max(dim=2)[0]
            # max_reg_targets_per_im = torch.min(lr_max, tb_max)
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(
                dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds
        }

    def locations_to_crit_box(self, locations, num_loc_list, xs, ys, reg_pred=None):
        loc_to_crit_box = []

        st = 0

        for l, stride in enumerate(self.strides):
            en = num_loc_list[l] + st
            loc_to_anchor_size = locations[st:en].new_tensor(stride)
            loc_to_anchor_size = loc_to_anchor_size[None].expand(num_loc_list[l])
            curr_xs = xs[st:en]
            curr_ys = ys[st:en]
            if reg_pred == None:
                anchor = cat([(curr_xs - loc_to_anchor_size / 2).unsqueeze(1), (curr_ys - loc_to_anchor_size / 2).unsqueeze(1), 
                                (curr_xs + loc_to_anchor_size / 2).unsqueeze(1) , (curr_ys + loc_to_anchor_size / 2).unsqueeze(1)], dim=1)
                loc_to_crit_box.append(anchor)
            else:
                curr_reg_pred = reg_pred[l].permute(1,2,0).reshape(-1, 4) * loc_to_anchor_size.unsqueeze(1)
                pred_box = cat([(curr_xs - curr_reg_pred[:,0]).unsqueeze(1), (curr_ys - curr_reg_pred[:,1]).unsqueeze(1),
                                   (curr_xs + curr_reg_pred[:,2]).unsqueeze(1), (curr_ys + curr_reg_pred[:,3]).unsqueeze(1)],dim=1)
                loc_to_crit_box.append(pred_box)
                
            st = en

        loc_to_crit_box = torch.cat(loc_to_crit_box, dim=0)

        return loc_to_crit_box

    def regression_positive_sample_seleciton(self, locations, num_loc_list, reg_pred, gt_boxes, pos_inds):
        """
        Regression positive sample selection part

        1. Calculate IOU between prediction box and given GT instances            
        2. Get mean and std of IOU per GT instances
        3. set positive index as ( is_in_box && IOU > mean + std * labmd_sched )
        """

        xs, ys = locations[:, 0], locations[:, 1]

        #calculate anchors based on locations. size of anchor is based on stride
        loc_to_crit_box = self.locations_to_crit_box(locations, num_loc_list, xs, ys, 
                                reg_pred if reg_pred != None else None)

        pre_calc_IoU = pairwise_giou(Boxes(loc_to_crit_box), gt_boxes)

        iou_thr = []
        in_boxes = pos_inds.nonzero()
        
        for i in range(pre_calc_IoU.shape[1]):
            per_idx = in_boxes[in_boxes[:,1] == i, 0]
            mean = pre_calc_IoU[per_idx, i].mean()
            std = pre_calc_IoU[per_idx, i].std()
            if len(per_idx) > 1:
                iou_thr = mean + std * self.positive_sample_rate
            else:
                iou_thr = 0.0

            if (pre_calc_IoU[:,i] >= iou_thr).sum() == 0:
                iou_thr = 0.0

            prev_pos = pos_inds[:,i].sum()

            pos_inds[:,i]*=(pre_calc_IoU[:,i] >= iou_thr)
            self.PIoU_thr+=iou_thr

            post_pos = pos_inds[:,i].sum()

            self.RPSR+=(post_pos / (prev_pos + 1e-6))
        
        return pos_inds, pre_calc_IoU

    def classification_positive_sample_seleciton(self, curr_classes, logits_pred, pos_inds):
        """
        Regression positive sample selection part

        1. Get mean and std of classification score per GT instances
        2. set positive index as ( is_in_box && IOU > mean + std * positive sample rate )
        """

        pairwise_cls = cat(logits_pred, dim=0).sigmoid().reshape(-1, self.num_classes)[:,curr_classes]

        in_boxes = pos_inds.nonzero()
        

        for i in range(pairwise_cls.shape[1]):
            per_idx = in_boxes[in_boxes[:,1] == i, 0]
            mean = pairwise_cls[per_idx, i].mean()
            std = pairwise_cls[per_idx, i].std()
            if len(per_idx) > 1:
                cls_thr = mean + std * self.positive_sample_rate
            else:
                cls_thr = 0.0
            if (pairwise_cls[:,i] >= cls_thr).sum() == 0:
                cls_thr = 0.0

            prev_pos = pos_inds[:,i].sum()
            pos_inds[:,i]*=(pairwise_cls[:,i] >= cls_thr)
            self.PCLS_thr+=cls_thr
        
            post_pos = pos_inds[:,i].sum()

            self.CPSR += (post_pos / (prev_pos + 1e-6))

        return pos_inds


    def compute_targets_ATSS(self, locations, targets, size_ranges, num_loc_list, logits_pred=None, reg_pred=None):
        labels = []
        reg_targets = []
        iou_targets = []
        cid_targets = []
        rid_targets = []
        #target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_objects = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                #target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            """
            Prepare area and is_in_box for usage in RPSS and CPSS
            """

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, num_loc_list, xs, ys,
                    bitmasks=bitmasks, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            rpos_inds, iou_trg = self.regression_positive_sample_seleciton(locations, num_loc_list, 
                [reg[im_i].detach().clone() for reg in reg_pred], 
                targets_per_im.gt_boxes, copy.deepcopy(is_in_boxes))

            cpos_inds = self.classification_positive_sample_seleciton(targets_per_im.gt_classes,
                [logit[im_i].detach().clone().reshape(self.num_classes,-1).t() for logit in logits_pred], 
                copy.deepcopy(is_in_boxes))

            locations_to_gt_area = area[None].repeat(len(locations), 1)

            lr_max = reg_targets_per_im[:,:,[0,2]].max(dim=2)[0]
            tb_max = reg_targets_per_im[:,:,[1,3]].max(dim=2)[0]

            min_reg_targets_per_im = torch.min(lr_max, tb_max)
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

            # limit the regression range for each location
            weird_in_the_level = \
                (min_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (min_reg_targets_per_im <= size_ranges[:, [1]])

            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            is_cared_in_the_level |= weird_in_the_level


            #target_inds_per_im = locations_to_gt_inds + num_targets

            """
            Solve multiple target anghor problem by using smallest area first policy
            and apply positive sample index of both classification and regression
            """

            def solve_multiple_target(locations_to_gt_area, is_cared_in_the_level, pos_inds):
                locations_to_gt_area[pos_inds == 0] = INF
                locations_to_gt_area[is_cared_in_the_level == 0] = INF
                locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

                return locations_to_min_area, locations_to_gt_inds

            cpos_to_min_area, cpos_to_gt_inds = solve_multiple_target(copy.deepcopy(locations_to_gt_area), 
                    is_cared_in_the_level, cpos_inds)
            rpos_to_min_area, rpos_to_gt_inds = solve_multiple_target(copy.deepcopy(locations_to_gt_area), 
                    is_cared_in_the_level, rpos_inds)

            labels_per_im = labels_per_im[cpos_to_gt_inds]
            labels_per_im[cpos_to_min_area == INF] = self.num_classes

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), rpos_to_gt_inds]
            iou_targets_per_im = iou_trg[range(len(locations)), rpos_to_gt_inds]
            iou_targets_per_im[iou_targets_per_im < 1] += 1e-6
            iou_targets_per_im[rpos_to_min_area == INF] = 0

            cpos_to_gt_inds += num_objects
            rpos_to_gt_inds += num_objects

            cpos_to_gt_inds[cpos_to_min_area == INF] = -1
            rpos_to_gt_inds[rpos_to_min_area == INF] = -1
            

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            iou_targets.append(iou_targets_per_im)
            cid_targets.append(cpos_to_gt_inds)
            rid_targets.append(rpos_to_gt_inds)
            #target_inds.append(target_inds_per_im)
            num_objects += len(targets_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "iou_targets": iou_targets,
            "cid_targets": cid_targets,
            "rid_targets": rid_targets,
            #"target_inds": target_inds
        }, num_objects

    def losses(self, logits_pred, reg_pred, cid_pred, rid_pred, iou_pred, locations, gt_instances, relation_net, top_feats=None):
        """
        Return the losses from a set of ADCR predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        # initialize statistics
        self.RPSR = 0
        self.CPSR = 0
        self.PIoU_thr = 0
        self.PCLS_thr = 0

        training_targets, num_objects = self._get_ground_truth(locations, gt_instances, logits_pred, reg_pred)

        if self.positive_sample_rate < 0.5:
            self.positive_sample_rate += self.pss_diff

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        instances.labels = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["labels"]
        ], dim=0)
        """
        instances.gt_inds = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.reshape(-1) for x in training_targets["target_inds"]
        ], dim=0)
        """
        instances.im_inds = cat([
            x.reshape(-1) for x in training_targets["im_inds"]
        ], dim=0)
        instances.reg_targets = cat([
            # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
            x.reshape(-1, 4) for x in training_targets["reg_targets"]
        ], dim=0,)

        instances.cid_targets = cat([
            # Reshape: (N, Hi, Wi, 1) -> (N*Hi*Wi)
            x.reshape(-1) for x in training_targets["cid_targets"]
        ], dim=0,)
        instances.rid_targets = cat([
            # Reshape: (N, Hi, Wi, 1) -> (N*Hi*Wi)
            x.reshape(-1) for x in training_targets["rid_targets"]
        ], dim=0,)
        instances.iou_targets = cat([
            # Reshape: (N, Hi, Wi, 1) -> (N*Hi*Wi)
            x.reshape(-1) for x in training_targets["iou_targets"]
        ], dim=0,)

        instances.locations = cat([
            x.reshape(-1, 2) for x in training_targets["locations"]
        ], dim=0)
        instances.fpn_levels = cat([
            x.reshape(-1) for x in training_targets["fpn_levels"]
        ], dim=0)

        instances.logits_pred = cat([
            # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
            x.permute(0, 2, 3, 1).reshape(-1, self.num_classes) for x in logits_pred
        ], dim=0,)
        instances.reg_pred = cat([
            # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
            x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred
        ], dim=0,)
        instances.iou_pred = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
            x.permute(0, 2, 3, 1).reshape(-1) for x in iou_pred
        ], dim=0,)
        instances.cid_pred = cat([
            # Reshape: (N, D, Hi, Wi) -> (N*Hi*Wi,D)
            x.permute(0, 2, 3, 1).reshape(-1, self.emb_dim) for x in cid_pred
        ], dim=0,)
        instances.rid_pred = cat([
            # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,D)
            x.permute(0, 2, 3, 1).reshape(-1, self.emb_dim) for x in rid_pred
        ], dim=0,)

        if len(top_feats) > 0:
            instances.top_feats = cat([
                # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in top_feats
            ], dim=0,)

        return self.adcr_losses(instances, num_objects, relation_net)

    def adcr_losses(self, instances, num_objects, relation_net):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        # prepare num_pos_avgs

        labels = instances.labels.flatten()
        piou = instances.iou_targets.flatten()

        cpos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        rpos_inds = torch.nonzero(piou != 0).squeeze(1)

        num_cpos_local = cpos_inds.numel()
        num_rpos_local = rpos_inds.numel()

        num_gpus = get_world_size()

        total_num_cpos = reduce_sum(cpos_inds.new_tensor([num_cpos_local])).item()
        total_num_rpos = reduce_sum(rpos_inds.new_tensor([num_rpos_local])).item()

        num_cpos_avg = max(total_num_cpos / num_gpus, 1.0)
        num_rpos_avg = max(total_num_rpos / num_gpus, 1.0)

        # classification loss

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[cpos_inds, labels[cpos_inds]] = 1

        class_loss = sigmoid_focal_loss_jit(
            instances.logits_pred,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_cpos_avg

        if self.focal_piou:
            piou_diff = (instances.iou_pred.sigmoid() - instances.iou_targets) ** 2
            piou_loss = (-self.focal_loss_alpha * (piou_diff) ** self.focal_loss_gamma * (1 - piou_diff).log()).sum() / num_rpos_avg
        else:
            piou_loss = F.mse_loss(instances.iou_pred.sigmoid(), instances.iou_targets,
                               reduction="sum") / num_rpos_avg
        # regression loss
         
        reg_pos_instances = instances[rpos_inds]
        reg_pos_instances.pos_inds = rpos_inds

        if rpos_inds.numel() > 0:
            reg_loss = self.loc_loss_func(
                reg_pos_instances.reg_pred,
                reg_pos_instances.reg_targets,
                #ctrness_targets
            ).mean()
            #class_loss[pos_inds, labels[pos_inds]] *= (1 - reg_loss/2).detach()
            #class_loss[pos_inds] *= (1 - reg_loss/2).unsqueeze(1).detach()
            #reg_loss = reg_loss.sum() / loss_denorm


        else:
            reg_loss = instances.reg_pred.sum() * 0
            piou_loss = instances.iou_pred.sum() * 0
        
        # embedding loss (cid, rid)

        emb_loss = self.embedding_loss(instances[rpos_inds], instances[cpos_inds], relation_net)

        losses = {
            "loss_adcr_cls": class_loss,
            "loss_adcr_loc": reg_loss,
            "loss_adcr_piou": piou_loss,
            "loss_adcr_emb": emb_loss,
        }
        extras = {
            "instances": instances,
            "num_objects": num_objects
        }
        return extras, losses
    
    def embedding_loss(self, rpos_instances, cpos_instances, relation_net):
        """
        1. get mean embedding vector per object
        2. calculate gather loss + farther loss
        """

        pred_emb = cat([rpos_instances.rid_pred, cpos_instances.cid_pred], dim=0)
        target_id = cat([rpos_instances.rid_targets, cpos_instances.cid_targets], dim=0)
        unique_id = torch.arange(len(target_id.unique()), device=target_id.device)
        for l, uid in enumerate(target_id.unique()):
            target_id[target_id==uid] = unique_id[l]


        C = self.emb_dim
        object_proto = torch.zeros(len(unique_id), C, device=pred_emb.device)

        for i in range(len(unique_id)):
            object_group = pred_emb[target_id == unique_id[i]]
            object_proto[i] = object_group.mean(dim=0)
        
        
        feature = pred_emb.unsqueeze(1).repeat(1,len(unique_id),1) - object_proto.unsqueeze(0).repeat(len(pred_emb),1,1)
        logits_pred = relation_net(feature.transpose(1,2)).squeeze(1)

        one_hot_target = torch.zeros_like(logits_pred)
        one_hot_target[range(len(target_id)), target_id] = 1

        emb_loss = sigmoid_focal_loss_jit(
            logits_pred,
            one_hot_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="mean",
        )
        return emb_loss



    def predict_proposals(
            self, logits_pred, reg_pred, cid_pred, rid_pred, iou_pred,
            locations, image_sizes, relation_net, top_feats=None
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_iou_thresh = self.pre_nms_iou_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_iou_thresh = self.pre_nms_iou_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locations, "o": logits_pred,
            "r": reg_pred, "pi": iou_pred, "ci": cid_pred, "ri": rid_pred,
            "s": self.strides, "pooler": self.pooler.level_poolers
        }

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            r = per_bundle["r"] * per_bundle["s"]
            s = per_bundle["s"]
            pi = per_bundle["pi"]
            ci = per_bundle["ci"]
            ri = per_bundle["ri"]
            pooler = per_bundle["pooler"]

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, pi, ci, ri, s, image_sizes, relation_net, pooler
                )
            )

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = l.new_ones(
                    len(per_im_sampled_boxes), dtype=torch.long
                ) * i

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map(
            self, locations, logits_pred, reg_pred, iou_pred,
            cid_pred, rid_pred, stride, image_sizes, relation_net, pooler
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        logits_pred = logits_pred.sigmoid()
        
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        
        iou_pred = iou_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        iou_pred = iou_pred.reshape(N, -1)#.sigmoid()

        cid_pred = cid_pred.view(N, self.emb_dim, H, W).permute(0, 2, 3, 1)
        cid_pred = cid_pred.reshape(N, -1, self.emb_dim)

        rid_pred = rid_pred.reshape(N, self.emb_dim, H, W).permute(0, 2, 3, 1)
        rid_pred = rid_pred.reshape(N, -1, self.emb_dim)

        # we first filter detection result with lower than iou_threshold
        candidate_inds = (iou_pred > self.pre_nms_iou_thresh) * (logits_pred.reshape(N, -1, C).max(dim=2)[0] > self.pre_nms_thresh)
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        results = []
        for i in range(N):
            # Match box regression and classificatoin
            # currently use voting system according to covered area of box in here
            per_box_cls = logits_pred[i]
            per_candidate_inds = candidate_inds[i]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]

            per_box_piou = iou_pred[i]
            per_box_piou = per_box_piou[per_box_loc]

            per_locations = locations[per_box_loc]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            per_box_cls, per_class = self.match_reg_cls(detections, per_box_cls, pooler) 

            per_pre_nms_top_n = pre_nms_top_n[i]
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                # sort prediction 1) iou  / 2) cls score and take top_k
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_box_piou = per_box_piou[top_k_indices]
                detections = detections[top_k_indices]

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            boxlist.centerness = per_box_piou
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def match_reg_cls(self, detection, logits_pred, pooler):
        batch_id = torch.zeros((len(detection),1), device=detection.device)
        pooled_cls = pooler(logits_pred.transpose(0,2).unsqueeze(0), cat([batch_id, detection], dim=1)).view(len(detection), self.num_classes)

        if len(detection) > 0:
            return pooled_cls.max(dim=1)
        else:
            return torch.zeros((0),device=detection.device), torch.zeros((0),device=detection.device)
