from logger import coil_logger
import torch.nn as nn
import torch
import importlib
import os

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Branching
from .building_blocks import FC

class CoILMemExtract(nn.Module):

    def __init__(self, params):

        super(CoILMemExtract, self).__init__()
        self.params = params
        number_first_layer_channels = 0

        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.ALL_FRAMES_INCLUDING_BLANK

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        self.predicted_speed = 0

        if 'res' in params['perception']:
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])
            self.perception = resnet_module(
                                    pretrained=g_conf.PRE_TRAINED,
                                    input_channels=number_first_layer_channels,
                                    num_classes=params['perception']['res']['num_classes']
                                )

            number_output_neurons = params['perception']['res']['num_classes']
                
        else:
            raise ValueError("perception type is not-defined")


        self.speed_branch = FC(
                                params={
                                    'neurons': [number_output_neurons] 
                                                + params['speed_branch']['fc']['neurons'] + [1],
                                    'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                    'end_layer': True
                                }
                            )

        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(
                FC(
                    params={
                        'neurons': [number_output_neurons]
                                    + params['branches']['fc']['neurons'] 
                                    + [len(g_conf.TARGETS)],
                        'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                        'end_layer': True
                    }
                )
            )
                        
        self.branches = Branching(branch_fc_vector)
        
        
    def forward(self, x):

        x, _ = self.perception(x)

        speed_branch_output = self.speed_branch(x)
        
        branch_outputs = self.branches(x)

        return branch_outputs + [speed_branch_output], x.detach()
