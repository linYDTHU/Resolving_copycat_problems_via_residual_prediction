import numpy as np
import scipy
import sys
import os
import glob
import torch
import cv2
import random
import time

from scipy.misc import imresize
from PIL import Image
from skimage import io

import matplotlib.pyplot as plt

try:
    from carla08 import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

from carla08.agent import CommandFollower
from carla08.client import VehicleControl

from network import CoILModel
from configs import g_conf
from logger import coil_logger
from process import ClientSideProcess, set_calibration, filte_out_near_nonplayer

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass




class CoILAgent(object):

    def __init__(self, checkpoint, town_name, carla_version='0.84'):

        # Set the carla version that is going to be used by the interface
        self._carla_version = carla_version
        self.checkpoint = checkpoint  # We save the checkpoint for some interesting future use.
        self._policy = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        self._mem_extract = CoILModel(g_conf.MEM_EXTRACT_MODEL_TYPE, g_conf.MEM_EXTRACT_MODEL_CONFIGURATION)
        self.first_iter = True

        # Load the model and prepare set it for evaluation
        self._policy.load_state_dict(checkpoint['policy_state_dict'])
        self._policy.cuda()
        self._policy.eval()
        self._mem_extract.load_state_dict(checkpoint['mem_extract_state_dict'])
        self._mem_extract.cuda()
        self._mem_extract.eval()

        self.latest_image = None
        self.latest_image_tensor = None

        if g_conf.USE_ORACLE or g_conf.USE_FULL_ORACLE:
            self.control_agent = CommandFollower(town_name)

        self.client_side_process = ClientSideProcess(g_conf.PREFRAME_PROCESS)

    def run_step(self, measurements, sensor_data, original_image_list, processed_image_list, directions, target, previous_actions_list=None, avoid_stop=True, perturb_speed=False):
        """
            Run a step on the benchmark simulation
        Args:
            measurements: The measurements
            sensor_data: The sensor data
            original_image_list: All the original images used on this benchmark, the input format is a list, including a series of continous frames.
            processed_image_list: All the processed images, the input format is a list, including a series of continous frames.
            directions: The directions, high level commands
            target: Final objective. Not used when the agent is predicting all outputs.
            previous_actions_list: All the previous actions used on this benchmark, optional

        Returns:
            Controls for the vehicle on the CARLA simulator.

        """
        # Take the forward speed and normalize it for it to go from 0-1
        norm_speed = measurements.player_measurements.forward_speed / g_conf.SPEED_FACTOR
        if perturb_speed and norm_speed < 0.01:
            norm_speed += random.uniform(0.05, 0.15)
        norm_speed = torch.cuda.FloatTensor([norm_speed]).unsqueeze(0)
        directions_tensor = torch.cuda.LongTensor([directions])

        input_tensor, original_image, processed_image = self._process_sensors(sensor_data, measurements,
                                                                              original_image_list, processed_image_list)

        # Compute the forward pass processing the sensors got from CARLA.
        obs_history = input_tensor
        measurement_input = torch.zeros_like(norm_speed)
        current_obs = torch.zeros_like(obs_history).cuda()
        current_obs[:, -3:] = obs_history[:, -3:]
        
        _, memory = self._mem_extract(obs_history)
        model_outputs = self._policy.forward_branch(current_obs, measurement_input, directions_tensor, memory)

        predicted_speed = self._policy.extract_predicted_speed()
        steer, throttle, brake = self._process_model_outputs(model_outputs[0], norm_speed, predicted_speed, avoid_stop)
        if self._carla_version == '0.9':
            import carla
            control = carla.VehicleControl()
        else:
            control = VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        # There is the posibility to replace some of the predictions with oracle predictions.
        if g_conf.USE_ORACLE:
            _, control.throttle, control.brake = self._get_oracle_prediction(
                measurements, target)

        if self.first_iter:
            coil_logger.add_message('Iterating', {"Checkpoint": self.checkpoint['iteration'],
                                                  'Agent': str(steer)},
                                    self.checkpoint['iteration'])
        self.first_iter = False

        return control, original_image, processed_image

    def get_attentions(self, layers=None):
        """

        Returns
            The activations obtained from the first layers of the latest iteration.

        """
        if layers is None:
            layers = [0, 1, 2]
        if self.latest_image_tensor is None:
            raise ValueError('No step was ran yet. '
                             'No image to compute the activations, Try Running ')
        all_layers = self._model.get_perception_layers(self.latest_image_tensor)
        cmap = plt.get_cmap('inferno')
        attentions = []
        for layer in layers:
            y = all_layers[layer]
            att = torch.abs(y).mean(1)[0].data.cpu().numpy()
            att = att / att.max()
            att = cmap(att)
            att = np.delete(att, 3, 2)
            attentions.append(imresize(att, [88, 200]))
        return attentions

    def _process_sensors(self, sensor_data, measurements, original_image_list, processed_image_list):

        if self._carla_version == '0.9':
            original_image = sensor_data['rgb']
        else:
            original_image = sensor_data['rgb'].data
        # sensor shape is (600, 800, 3)

        processed_image = self.client_side_process.process_func(original_image, measurements,
                                                                mask_probability=g_conf.PREFRAME_PROCESS_PROBABILITY)

        original_image_list.append(original_image)
        processed_image_list.append(processed_image)

        frame_sequence = processed_image_list[:g_conf.PREFRAME_PROCESS_NUM*g_conf.SEQUENCE_STRIDE:g_conf.SEQUENCE_STRIDE] + \
                         original_image_list[g_conf.PREFRAME_PROCESS_NUM*g_conf.SEQUENCE_STRIDE::g_conf.SEQUENCE_STRIDE]

        iteration = 0
        for sensor in frame_sequence:
            # io.imsave('./processed_image/frame_{}/{}.png'.format(frame_idx, iteration), sensor)

            sensor = sensor[g_conf.IMAGE_CUT[0]:g_conf.IMAGE_CUT[1], ...]
            sensor = scipy.misc.imresize(sensor, (g_conf.SENSORS['rgb'][1], g_conf.SENSORS['rgb'][2]))

            self.latest_image = sensor

            sensor = np.swapaxes(sensor, 0, 1)

            sensor = np.transpose(sensor, (2, 1, 0))

            sensor = torch.from_numpy(sensor / 255.0).type(torch.FloatTensor).cuda()

            if iteration == 0:
                image_input = sensor
            else:
                image_input = torch.cat((image_input, sensor), 0)

            iteration += 1

        if len(frame_sequence) != g_conf.ALL_FRAMES_INCLUDING_BLANK:
            # stack the blank frames
            if g_conf.BLANK_FRAMES_TYPE == 'black':
                image_input = torch.cat((torch.zeros((3*(g_conf.ALL_FRAMES_INCLUDING_BLANK - len(frame_sequence)),
                                                      g_conf.SENSORS['rgb'][1], g_conf.SENSORS['rgb'][2])).cuda(),
                                         image_input), 0)
            elif g_conf.BLANK_FRAMES_TYPE == 'copy':
                image_input = torch.cat((image_input[:3, ...].repeat(g_conf.ALL_FRAMES_INCLUDING_BLANK - len(frame_sequence), 1, 1),
                                         image_input), 0)

        image_input = image_input.unsqueeze(0)

        self.latest_image_tensor = image_input

        return image_input, original_image, processed_image

    def _process_model_outputs(self, outputs, norm_speed, predicted_speed, avoid_stop=True):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        assert len(g_conf.TARGETS) == len(outputs), 'the dimension of outputs does not match the TARGETS!'
        if len(g_conf.TARGETS) == 3:
            steer, throttle, brake = float(outputs[0]), float(outputs[1]), float(outputs[2])
            if brake < 0.05:
                brake = 0.0

            if throttle > brake:
                brake = 0.0
        elif len(g_conf.TARGETS) == 2:
            steer, throttle_brake = float(outputs[0]), float(outputs[1])
            if throttle_brake >= 0:
                throttle = throttle_brake
                brake = 0
            else:
                throttle = 0
                brake = -throttle_brake
        else:
            raise Exception('only support 2 or 3 dimensional outputs')

        if avoid_stop:
            real_speed = norm_speed * g_conf.SPEED_FACTOR
            real_predicted_speed = predicted_speed * g_conf.SPEED_FACTOR

            if real_speed < 5.0 and real_predicted_speed > 6.0:  # If (Car Stooped) and ( It should not have stoped)
                throttle += 20.0 / g_conf.SPEED_FACTOR - norm_speed
                brake = 0.0

        return steer, throttle, brake


    def _process_model_outputs_wp(self, outputs):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """
        # with waypoint
        wpa1, wpa2, throttle, brake = outputs[3], outputs[4], outputs[1], outputs[2]
        if brake < 0.2:
            brake = 0.0

        if throttle > brake:
            brake = 0.0

        steer = 0.7 * wpa2

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        return steer, throttle, brake

    def _get_oracle_prediction(self, measurements, target):
        # For the oracle, the current version of sensor data is not really relevant.
        control, _, _, _, _ = self.control_agent.run_step(measurements, [], [], target)

        return control.steer, control.throttle, control.brake
