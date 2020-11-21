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

        V_iou = (object_std_i + object_std_j) / (V_dist_norm + object_std_i + object_std_j + 1e-5)
        V_iou = torch.triu(V_iou, diagonal=1)
        
        assert V_iou.isnan().sum() == 0

        return  V_iou.mean()
        """
        object_arange = torch.arange(len(object_group))
        for i, j in itertools.combinations(object_arange, 2):
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
        """
        