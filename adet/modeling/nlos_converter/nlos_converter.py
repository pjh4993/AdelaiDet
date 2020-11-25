from collections import defaultdict
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F
import math

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from adet.modeling.nlos_converter.build import NLOS_CONVERTER_REGISTRY
from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations

@NLOS_CONVERTER_REGISTRY.register()
class conv_fc_nlos_converter(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        """
        super().__init__()
        head_configs = {"conv": cfg.MODEL.NLOS_CONVERTER.NUM_CONVS,
                         "fc": (cfg.MODEL.NLOS_CONVERTER.IN_FC_CHANNELS,
                                 cfg.MODEL.NLOS_CONVERTER.OUT_FC_CHANNELS)}

        norm = None if cfg.MODEL.NLOS_CONVERTER.NORM == "none" else cfg.MODEL.NLOS_CONVERTER.NORM
        self.int_conv_channel = cfg.MODEL.NLOS_CONVERTER.INT_CONV_CHANNEL
        self.in_features = cfg.MODEL.NLOS_CONVERTER.IN_FEATURES
        input_shape = [input_shape[f] for f in self.in_features]
        self.num_levels = len(input_shape)
        self.laser_grid = cfg.NLOS.LASER_GRID
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        assert self.int_conv_channel % 32 == 0 or norm == None

        conv_tower = []
        for _ in range(head_configs["conv"]):
            conv_layer = nn.Conv2d(in_channels, self.int_conv_channel, kernel_size=3, stride=1, padding=1, bias=True)
            in_channels = self.int_conv_channel
            conv_tower.append(conv_layer)
            if norm == "GN":
                conv_tower.append(nn.GroupNorm(32, in_channels))
            elif norm == "NaiveGN":
                conv_tower.append(NaiveGroupNorm(32, in_channels))

        self.add_module('{}_tower'.format("conv"), nn.Sequential(*conv_tower))

        fc_per_level = []
        for in_fc_channel, out_fc_channel, k in zip(head_configs["fc"][0], head_configs["fc"][1], self.in_features):
            fc_layer = nn.Linear(in_fc_channel * self.int_conv_channel * 2 * (self.laser_grid **2), out_fc_channel[0] * out_fc_channel[1] * self.int_conv_channel)
            torch.nn.init.xavier_uniform_(fc_layer.weight)
            torch.nn.init.constant_(fc_layer.bias, 0)
            fc_per_level.append(fc_layer)
            self.add_module('{}_fc_layer'.format(k),
                        fc_layer)

        for modules in [
            self.conv_tower
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.fc_per_level = fc_per_level
        self.head_configs = head_configs

    def forward(self, features: List[torch.Tensor]):
        """
        """

        converted_feature = dict()

        for k in self.in_features:
            converted_feature[k] = []

        for x in features:
            x = [x[f] for f in self.in_features]
            for k, v in zip(self.in_features, x):#, self.fc_per_level, self.head_configs["fc"][1]):
                t = F.relu(self.conv_tower(v)).reshape(1,-1)
                converted_feature[k].append(t)

        for k, fc, output_shape in zip(self.in_features, self.fc_per_level, self.head_configs["fc"][1]):
            t = torch.cat(converted_feature[k], dim=0)
            N = t.shape[0]
            converted_feature[k] = fc(t).reshape(N, self.int_conv_channel, output_shape[1], output_shape[0])
        return converted_feature

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.int_conv_channel, stride=0
            )
            for name in self.in_features
        }

@NLOS_CONVERTER_REGISTRY.register()
class channel_preserving_nlos_converter(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        """
        super().__init__()
        head_configs = {"conv": cfg.MODEL.NLOS_CONVERTER.NUM_CONVS,
                        "fc": (cfg.MODEL.NLOS_CONVERTER.IN_FC_CHANNELS,
                                cfg.MODEL.NLOS_CONVERTER.OUT_FC_CHANNELS)}
 
        norm = None if cfg.MODEL.NLOS_CONVERTER.NORM == "none" else cfg.MODEL.NLOS_CONVERTER.NORM
        self.int_conv_channel = cfg.MODEL.NLOS_CONVERTER.INT_CONV_CHANNEL
        self.in_features = cfg.MODEL.NLOS_CONVERTER.IN_FEATURES
        input_shape = [input_shape[f] for f in self.in_features]
        self.num_levels = len(input_shape)
        self.laser_grid = cfg.NLOS.LASER_GRID
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        assert self.int_conv_channel % 32 == 0 or norm == None

        conv_tower = []
        for _ in range(head_configs["conv"]):
            conv_layer = nn.Conv2d(in_channels, self.int_conv_channel, kernel_size=3, stride=1, padding=1, bias=True)
            in_channels = self.int_conv_channel
            conv_tower.append(conv_layer)
            if norm == "GN":
                conv_tower.append(nn.GroupNorm(32, in_channels))
            elif norm == "NaiveGN":
                conv_tower.append(NaiveGroupNorm(32, in_channels))

        self.add_module('{}_tower'.format("conv"),
                        nn.Sequential(*conv_tower))

        fc_per_level = []
        for laser_wh, out_fc_channel, k in zip(head_configs["fc"][0], head_configs["fc"][1], self.in_features):
            fc_layer = nn.Linear(laser_wh * (self.laser_grid **2), out_fc_channel[0] * out_fc_channel[1])
            torch.nn.init.xavier_uniform_(fc_layer.weight)
            torch.nn.init.constant_(fc_layer.bias, 0)
            fc_per_level.append(fc_layer)
            self.add_module('{}_fc_layer'.format(k),
                        fc_layer)

        for modules in [
            self.conv_tower
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        self.fc_per_level = fc_per_level
        self.head_configs = head_configs

    def forward(self, features: List[torch.Tensor]):
        """
        """
        grid_size = self.laser_grid ** 2
        
        converted_feature = dict()
         
        for k in self.in_features:
            converted_feature[k] = []
 
        for x in features:
            x = [x[f] for f in self.in_features]
            for k, v in zip(self.in_features, x):#, self.fc_per_level, self.head_configs["fc"][1]):
                t = F.relu(self.conv_tower(v))
                t = t[torch.arange(start=0, end=grid_size * 2, step=2)] + t[torch.arange(start=1, end=grid_size *2, step=2)]
                t = t.permute(1,0,2,3).reshape(1, self.int_conv_channel,-1)
                converted_feature[k].append(t)
 
        for k, fc, output_shape in zip(self.in_features, self.fc_per_level, self.head_configs["fc"][1]):
            t = torch.cat(converted_feature[k], dim=0)
            N = t.shape[0]
            converted_feature[k] = fc(t).reshape(N, self.int_conv_channel, output_shape[1], output_shape[0])
 
        return converted_feature
     
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.int_conv_channel, stride=0
            )
            for name in self.in_features
        }