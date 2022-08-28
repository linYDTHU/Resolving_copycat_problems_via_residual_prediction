from logger import coil_logger
import torch.nn as nn
import torch
import importlib
import os

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join


class CoILPolicy(nn.Module):

    def __init__(self, params):

        super(CoILPolicy, self).__init__()
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

        self.measurements = FC(
                                params={
                                    'neurons': [len(g_conf.INPUTS)] + params['measurements']['fc']['neurons'],
                                    'dropouts': params['measurements']['fc']['dropouts'],
                                    'end_layer': False
                                }
                            )


        self.join = Join(
                        params={
                            'after_process':
                                FC(
                                    params={
                                        'neurons':
                                               [params['measurements']['fc']['neurons'][-1]
                                                + number_output_neurons
                                                + params['memory_dim']] +
                                               params['join']['fc']['neurons'],
                                        'dropouts': params['join']['fc']['dropouts'],
                                        'end_layer': False
                                    }
                                ),
                            'mode': 'cat'
                        }
                    )


        self.speed_branch = FC(
                                params={
                                    'neurons': [number_output_neurons] + params['speed_branch']['fc']['neurons'] + [1],
                                    'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                    'end_layer': True
                                }
                            )


        branch_fc_vector = []
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(
                FC(
                    params={
                        'neurons': [params['join']['fc']['neurons'][-1]]
                                    + params['branches']['fc']['neurons']
                                    + [len(g_conf.TARGETS)],
                        'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                        'end_layer': True
                    }
                )
            )

        self.branches = Branching(branch_fc_vector)

    def forward(self, x, v, memory):
    
        x, _ = self.perception(x)

        m = self.measurements(v)

        m = torch.cat((m, memory), 1)
        j = self.join(x, m)
            
        branch_outputs = self.branches(j)
        
        speed_branch_output = self.speed_branch(x)

        return branch_outputs + [speed_branch_output]

    def forward_branch(self, x, v, branch_number, memory):
    
        output = self.forward(x, v, memory)
        self.predicted_speed = output[-1]
        control = output[0:4]
        output_vec = torch.stack(control)

        return self.extract_branch(output_vec, branch_number)

    def extract_branch(self, output_vec, branch_number):

        branch_number = command_number_to_index(branch_number)

        if len(branch_number) > 1:
            branch_number = torch.squeeze(branch_number.type(torch.cuda.LongTensor))
        else:
            branch_number = branch_number.type(torch.cuda.LongTensor)

        branch_number = torch.stack([branch_number,
                                     torch.cuda.LongTensor(range(0, len(branch_number)))])

        return output_vec[branch_number[0], branch_number[1], :]

    def extract_predicted_speed(self):
        # return the speed predicted in forward_branch()
        return self.predicted_speed