import logging
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import ProposalNetwork, GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
import torch
import torch
import gc


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
class OneStageDetector(ProposalNetwork):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Uses "instances" as the return key instead of using "proposal".
    """
    def forward(self, batched_inputs):
        if self.training:
            return super().forward(batched_inputs)
        processed_results = super().forward(batched_inputs)
        processed_results = [{"one_stage_instances": r["proposals"]} for r in processed_results]
        return processed_results

@META_ARCH_REGISTRY.register()
class MetaProposalNetwork(ProposalNetwork):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        """
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))
        """
        pass

    def image_to_tensor(self, batched_image):
        images = [x["image"].to(self.device) for x in batched_image]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    def extract_instance(self, batched_input, batched_label):
        if "instances" in batched_input[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_input]
        elif "targets" in batched_input[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_input]
        else:
            gt_instances = None

        masked_instances = []
        for instances in gt_instances:
            instance_label = instances.get_fields()["gt_classes"]
            instance_label = instance_label.to(batched_label.device)
            mask_label = [x in batched_label for x in instance_label]
            masked_instances.append(instances[mask_label])
        
        return masked_instances


    def split_by_set(self, batched_inputs, batched_labels):

        batched_images = []
        batched_features = []
        batched_gt_instances = []

        for batched_input, batched_label in zip(batched_inputs, batched_labels):
            sup_images = self.image_to_tensor(batched_input[0])
            que_images = self.image_to_tensor(batched_input[1])

            batched_images.append([sup_images, que_images])
            batched_features.append([self.backbone(sup_images.tensor), self.backbone(que_images.tensor)])

            sup_masked_instances = self.extract_instance(batched_input[0], batched_label)
            que_masked_instances = self.extract_instance(batched_input[1], batched_label)

            batched_gt_instances.append({
                'instances':[sup_masked_instances, que_masked_instances],
                'labels' : batched_label
            })
        
        return batched_images, batched_features, batched_gt_instances

    def forward(self, batched_classwise_inputs):
        #stage 1. extract class, box feature from images

        #stage 2. find prototype for each class & box

        #stage 3. calculate loss_cls, loss_reg from query based on prototype

        """
        batched_input :
            support set : list of dict which contain image info, image tensor, annotation info
            query set : list of dict of query
            labels : current class label of interest
        """

        batched_inputs = []
        batched_labels = []
        for batch in batched_classwise_inputs:
            batched_inputs.append([batch["support_set"], batch["query_set"]])
            batched_labels.append(batch["labels"])

        batched_images, batched_features, batched_gt_instances = self.split_by_set(batched_inputs, batched_labels)

        batched_proposals, proposal_losses = self.proposal_generator(batched_images, batched_features, batched_gt_instances)

        if self.training:
            return proposal_losses

        batched_processed_results = []
        for proposals, inputs, images in zip(batched_proposals, batched_inputs, batched_images):
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                proposals, inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            batched_processed_results.append(processed_results)

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

        return batched_processed_results

def build_top_module(cfg):
    top_type = cfg.MODEL.TOP_MODULE.NAME
    if top_type == "conv":
        inp = cfg.MODEL.FPN.OUT_CHANNELS
        oup = cfg.MODEL.TOP_MODULE.DIM
        top_module = nn.Conv2d(
            inp, oup,
            kernel_size=3, stride=1, padding=1)
    else:
        top_module = None
    return top_module


@META_ARCH_REGISTRY.register()
class OneStageRCNN(GeneralizedRCNN):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.top_module = build_top_module(cfg)
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances, self.top_module)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(
                    images, features, None, self.top_module)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return OneStageRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
