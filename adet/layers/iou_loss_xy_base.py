import torch
from torch import nn


class IOULossXYbase(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """
        pred_x = pred[:, 0]
        pred_y = pred[:, 1]
        pred_width = pred[:, 2]
        pred_height = pred[:, 3]

        target_x = target[:, 0]
        target_y = target[:, 1]
        target_width = target[:, 2]
        target_height = target[:, 3]

        target_aera = target_width * target_height
        pred_aera = pred_width * pred_height

        min_base_x = torch.min(0, torch.abs(torch.min(pred_x - pred_width * 0.5, target_x - target_width*0.5).detach()))
        min_base_y = torch.min(0, torch.abs(torch.min(pred_y - pred_height * 0.5, target_y - target_height*0.5).detach()))

        pred_left = pred_x + min_base_x - pred_width * 0.5
        pred_right = pred_x + min_base_x + pred_width * 0.5
        pred_top = pred_y + min_base_y - pred_height * 0.5
        pred_bottom = pred_x + min_base_x + pred_height * 0.5

        target_left = target_x + min_base_x - target_width * 0.5
        target_right = target_x + min_base_x + target_width * 0.5
        target_bottom = target_y + min_base_y - target_height * 0.5
        target_top = target_y + min_base_y + target_height * 0.5

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)*(1-ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        elif self.loc_loss_type == 'giou_focal':
            losses = -torch.log((1 + gious)/2)*((1 - gious)/2)
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()
