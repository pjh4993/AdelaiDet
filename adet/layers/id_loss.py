import torch
from torch import nn


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
        
        # loss V_ratio : increase v_ratio of id vector

        object_group = []
        for i in gt_id:
            object_group.append(target_id == i)
        
        object_group.append(torch.zeros_like(object_group[-1]))
        D_ratio = []

        for i in range(len(object_group)):
            for j in range(i+1, len(object_group)):
                group_i = pred_id[object_group[i]]
                group_j = pred_id[object_group[j]]

                if object_group[j].sum() == 0:
                    group_j = torch.zeros(2, device=group_i.device) 
                
                proto_i = group_i.mean()
                proto_j = group_j.mean()

                V_dist = proto_j - proto_i
                V_dist_norm = torch.abs(torch.norm(V_dist,p=2,dim=0))
                V_dist = V_dist.div(V_dist_norm.expand_as(V_dist)).detach()

                V_intra_i = torch.abs(group_i.std() * V_dist) if ~group_i.std().isnan() else torch.zeros(1, device=group_i.device).item()
                V_intra_j = torch.abs(group_j.std() * V_dist) if ~group_j.std().isnan() else torch.zeros(1, device=group_j.device).item()

                V_iou = (V_intra_i + V_intra_j) / (V_dist_norm + V_intra_i + V_intra_j)

                assert ~V_iou.isnan()
                D_ratio.append(V_iou)
               
        return torch.stack(D_ratio).mean()
        