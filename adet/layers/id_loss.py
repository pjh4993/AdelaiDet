import torch
from torch import nn
import itertools


class IDLoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, id_loss_type='.', hard_threshold=0.1):
        super(IDLoss, self).__init__()
        self.loss_type = id_loss_type
        self.hard_threshold = hard_threshold

    def forward(self, pred_id, target_id, weight=None):
        """
        Args:
            pred: Nx4 predicted bounding boxes
            target: Nx4 target bounding boxes
            weight: N loss weight for each instance
        """

        gt_id = target_id.unique()
        object_num = len(gt_id) + 1

        # loss V_ratio : increase v_ratio of id vector

        C = pred_id.shape[-1]
        object_proto = torch.zeros(object_num, C, device=pred_id.device)
        object_std = torch.zeros(object_num, 1, device=pred_id.device)

        for i in range(len(gt_id)):
            object_group = pred_id[target_id == gt_id[i]]
            object_proto[i] = object_group.mean()
            object_std[i] = object_group.std() if object_group.shape[0] > 1 else 0.0

        object_proto = object_proto.unsqueeze(1)
        V_dist = (object_proto.permute(1,0,2) - object_proto)
        V_dist_norm = torch.abs(torch.norm(V_dist,p=2,dim=2))
        V_dist_detach = V_dist.div(V_dist_norm.unsqueeze(-1).expand_as(V_dist) + 1e-5).detach()

        object_std_i = torch.abs(object_std.repeat(1,object_num) * V_dist_detach)
        object_std_j = torch.abs(object_std.transpose(1,0).repeat(object_num,1) * V_dist_detach)

        V_iou = (object_std_i + object_std_j+ 1 - V_dist_norm) / (V_dist_norm + object_std_i + object_std_j + 1e-5 + 1)
        pos_ind = torch.triu_indices(object_num, object_num, offset=1)
        V_iou = V_iou.mean(dim=-1)[pos_ind[1], pos_ind[0]]
        assert V_iou.isnan().sum() == 0

        V_iou = - (V_iou) * torch.log(1 - V_iou)
        return  V_iou.mean()
