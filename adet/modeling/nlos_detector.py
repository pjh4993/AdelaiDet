import logging
from torch import nn
import torch

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import ProposalNetwork, GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from adet.modeling.nlos_converter import build_nlos_converter

def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for 
    bezier control points.
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)

    return results


@META_ARCH_REGISTRY.register()
class NLOSDetector(ProposalNetwork):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Uses "instances" as the return key instead of using "proposal".
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.laser_grid = cfg.NLOS.LASER_GRID
        self.nlos_converter = build_nlos_converter(cfg, self.backbone.output_shape())
        self.proposal_generator = build_proposal_generator(cfg, self.nlos_converter.output_shape())

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        laser_image_groups = [x["laser_images"].to(self.device) for x in batched_inputs]
        laser_image_groups = [((x - self.pixel_mean) / self.pixel_std) for x in laser_image_groups]

        gt_images = [x["gt_image"].to(self.device) for x in batched_inputs]

        laser_grid_batch = []
        for laser_images in laser_image_groups:
            single_laser_image = ImageList.from_tensors([laser_images], self.backbone.size_divisibility)
            features = self.backbone(single_laser_image.tensor.squeeze(0))
            #_, H, W, C = features[0].shape
            #features = {k : v.reshape(2, self.laser_grid, self.laser_grid, H, W, C) for k, v in features.items()}
            laser_grid_batch.append(features)
 
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        #some network for generate feature for proposal generator

        detection_features = self.nlos_converter(laser_grid_batch)

        proposals, proposal_losses = self.proposal_generator(gt_images, detection_features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, gt_images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results