import os
import glob
import traceback
import collections
import sys
import math
import copy
import json
import random
import numpy as np
from queue import Queue

import torch
import cv2

from torch.utils.data import Dataset

from . import splitter
from . import data_parser

from configs import g_conf

from coilutils.general import sort_nicely



def parse_remove_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'remove'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key

    return name, conf_dict


def get_episode_weather(episode):
    with open(os.path.join(episode, 'metadata.json')) as f:
        metadata = json.load(f)
    print(" WEATHER OF EPISODE ", metadata['weather'])
    return int(metadata['weather'])


class CoILDataset(Dataset):
    """ The conditional imitation learning dataset"""

    def __init__(self, root_dir, transform=None, preload_name=None):
        # Setting the root directory for this dataset
        self.root_dir = root_dir

        self.img_h = g_conf.SENSORS['rgb'][1]  # generally is 88
        self.img_w = g_conf.SENSORS['rgb'][2]  # generally is 200

        if g_conf.PREFRAME_PROCESS != 'None':
            self.processed_root_dir = self.root_dir + '_' + g_conf.PREFRAME_PROCESS
        else:
            self.processed_root_dir = self.root_dir

        # We add to the preload name all the remove labels
        if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
            name, self._remove_params = parse_remove_configuration(g_conf.REMOVE)
            self.preload_name = preload_name + '_' + name
            self._check_remove_function = getattr(splitter, name)
        else:
            self._check_remove_function = lambda _, __: False
            self._remove_params = []
            self.preload_name = preload_name

        if g_conf.DATA_USED != 'all':
            self.preload_name += '_{}'.format(g_conf.DATA_USED)

        if g_conf.ASSIGN_WEATHER is not None and g_conf.ASSIGN_WEATHER is not "None":
            self.preload_name += '_weather{}'.format(g_conf.ASSIGN_WEATHER)

        print("preload Name ", self.preload_name)

        if g_conf.DATA_USED == 'all':
            self.previous_frame_action_step = 3  # the step is 3 because the image is save as [CentralRGB0, LeftRGB0, RightRGB0, CentralRGB1, LeftRGB1, RightRGB1...]
        elif g_conf.DATA_USED == 'central':
            self.previous_frame_action_step = 1
        elif g_conf.DATA_USED == 'side':
            self.previous_frame_action_step = 2
        else:
            raise NotImplementedError

        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
            print(" Loading from NPY ")
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'))
            print(self.sensor_data_names)
        else:
            self.sensor_data_names, self.measurements = self._pre_load_image_folders(root_dir)

        for idx in range(len(self.measurements)):
            if self.measurements[idx]['throttle'] >= self.measurements[idx]['brake']:
                self.measurements[idx]['throttle_brake'] = self.measurements[idx]['throttle']
            else:
                self.measurements[idx]['throttle_brake'] = -self.measurements[idx]['brake']

        print("preload Name ", self.preload_name)

        self.processed_num = min(g_conf.PREFRAME_PROCESS_NUM, g_conf.NUMBER_FRAMES_FUSION)

        self.transform = transform
        self.batch_read_number = 0

    def __len__(self):
        # if the input rgb is the stack of several consequent frames,
        # the length should be shorter than self.measuremets
        return len(self.measurements) - g_conf.NUMBER_FRAMES_FUSION + 1

    def __getitem__(self, index):
        """
        Get item function used by the dataset loader
        returns all the measurements with the desired image.

        Args:
            index:

        Returns:

        """
        try:
            if self.processed_num == g_conf.NUMBER_FRAMES_FUSION:
                img_path_list = [os.path.join(self.processed_root_dir,
                                              self.sensor_data_names[index].split('/')[-2],
                                              self.sensor_data_names[index].split('/')[-1])]
            else:
                img_path_list = [os.path.join(self.root_dir,
                                              self.sensor_data_names[index].split('/')[-2],
                                              self.sensor_data_names[index].split('/')[-1])]

            previous_frame_index = index
            previous_frame_index_buffer = index
            while len(img_path_list) < g_conf.NUMBER_FRAMES_FUSION:
                if (g_conf.NUMBER_FRAMES_FUSION - len(img_path_list)) <= self.processed_num:
                    root_dir_loading = self.processed_root_dir
                else:
                    root_dir_loading = self.root_dir

                previous_frame_index -= self.previous_frame_action_step
                if (previous_frame_index < 0) or (self.sensor_data_names[previous_frame_index].split('/')[0] != self.sensor_data_names[index].split('/')[0]):
                    img_path_list.append(os.path.join(root_dir_loading,
                                                      self.sensor_data_names[previous_frame_index_buffer].split('/')[-2],
                                                      self.sensor_data_names[previous_frame_index_buffer].split('/')[-1]))
                else:
                    img_path_list.append(os.path.join(root_dir_loading,
                                                      self.sensor_data_names[previous_frame_index].split('/')[-2],
                                                      self.sensor_data_names[previous_frame_index].split('/')[-1]))
                    previous_frame_index_buffer = previous_frame_index

            img_path_list.reverse()
            img_list = [cv2.imread(path, cv2.IMREAD_COLOR) for path in img_path_list]

            # Apply the image transformation
            for i in range(len(img_list)):
                img = img_list[i]
                if self.transform is not None:
                    boost = 1
                    img = self.transform(self.batch_read_number * boost, img)
                else:
                    img = img.transpose(2, 0, 1)
                img_list[i] = img

            if g_conf.BLANK_FRAMES_TYPE == 'black':
                blank_img_list = [np.zeros_like(img_list[0])
                                  for _ in range(g_conf.ALL_FRAMES_INCLUDING_BLANK - g_conf.NUMBER_FRAMES_FUSION)]
            elif g_conf.BLANK_FRAMES_TYPE == 'copy':
                blank_img_list = [img_list[0]
                                  for _ in range(g_conf.ALL_FRAMES_INCLUDING_BLANK - g_conf.NUMBER_FRAMES_FUSION)]
            else:
                raise ValueError('BLANK_FRAMES_TYPE must be black or copy')

            img_list = blank_img_list + img_list

            # test the code by saving the intermedia processed image
            # self.test(img_list, index)

            img_stack = np.vstack(img_list)
            img_stack = img_stack.astype(np.float)
            img_stack = torch.from_numpy(img_stack).type(torch.FloatTensor)
            img_stack = img_stack / 255.

            if index < 3 or (self.sensor_data_names[index-3].split('/')[0] != self.sensor_data_names[index].split('/')[0]):
                action_index = index
            else:
                action_index = index - 3

            previous_directions = None
            previous_actions_list = []
            previous_action_index = action_index
            previous_action_index_buffer = action_index
            while len(previous_actions_list) < g_conf.NUMBER_PREVIOUS_ACTIONS*2:
                previous_action_index -= self.previous_frame_action_step
                if (previous_action_index < 0) or (self.sensor_data_names[previous_action_index].split('/')[0] != self.sensor_data_names[index].split('/')[0]):
                    previous_actions_list.append(self.measurements[previous_action_index_buffer]['throttle_brake'])
                    previous_actions_list.append(self.measurements[previous_action_index_buffer]['steer'])
                    if previous_directions is None:
                        previous_directions = self.measurements[previous_action_index_buffer]['directions']
                else:
                    previous_actions_list.append(self.measurements[previous_action_index_buffer]['throttle_brake'])
                    previous_actions_list.append(self.measurements[previous_action_index]['steer'])
                    if previous_directions is None:
                        previous_directions = self.measurements[previous_action_index]['directions']
                    previous_action_index_buffer = previous_action_index
            previous_actions_list.reverse()  # [..., action[-2]_steer, action[-2]_throttle, action[-2]_brake, action[-1]_steer, action[-1]_throttle, action[-1]_brake]
            previous_actions_stack = torch.from_numpy(np.array(previous_actions_list).astype(np.float)).type(torch.FloatTensor)

            measurements = self.measurements[index].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()

            measurements['rgb'] = img_stack
            if g_conf.NUMBER_PREVIOUS_ACTIONS > 0:
                measurements['previous_actions'] = previous_actions_stack
            else:
                if len(g_conf.TARGETS) == 3:
                    measurements['previous_actions'] = torch.from_numpy(np.zeros(3)).type(torch.FloatTensor)
                else:
                    measurements['previous_actions'] = torch.from_numpy(np.zeros(2)).type(torch.FloatTensor)

            measurements['previous_directions'] = torch.from_numpy(np.asarray([previous_directions, ]))

            self.batch_read_number += 1
        except AttributeError:
            print ("Blank IMAGE")

            measurements = self.measurements[0].copy()
            for k, v in measurements.items():
                v = torch.from_numpy(np.asarray([v, ]))
                measurements[k] = v.float()
            measurements['steer'] = 0.0
            measurements['throttle'] = 0.0
            measurements['brake'] = 0.0
            measurements['rgb'] = np.zeros((3*g_conf.NUMBER_FRAMES_FUSION, self.img_h, self.img_w))
            if len(g_conf.TARGETS) == 3:
                measurements['previous_actions'] = np.zeros(3*g_conf.NUMBER_PREVIOUS_ACTIONS)
            else:
                measurements['previous_actions'] = np.zeros(2*g_conf.NUMBER_PREVIOUS_ACTIONS)

        return measurements

    def is_measurement_partof_experiment(self, measurement_data):

        # If the measurement data is not removable is because it is part of this experiment dataa
        return not self._check_remove_function(measurement_data, self._remove_params)

    def _get_final_measurement(self, speed, measurement_data, angle,
                               directions, avaliable_measurements_dict):
        """
        Function to load the measurement with a certain angle and augmented direction.
        Also, it will choose if the brake is gona be present or if acceleration -1,1 is the default.

        Returns
            The final measurement dict
        """
        if angle != 0:
            measurement_augmented = self.augment_measurement(copy.copy(measurement_data), angle,
                                                             3.6 * speed,
                                                 steer_name=avaliable_measurements_dict['steer'])
        else:
            # We have to copy since it reference a file.
            measurement_augmented = copy.copy(measurement_data)

        if 'gameTimestamp' in measurement_augmented:
            time_stamp = measurement_augmented['gameTimestamp']
        else:
            time_stamp = measurement_augmented['elapsed_seconds']

        final_measurement = {}
        # We go for every available measurement, previously tested
        # and update for the measurements vec that is used on the training.
        for measurement, name_in_dataset in avaliable_measurements_dict.items():
            # This is mapping the name of measurement in the target dataset
            final_measurement.update({measurement: measurement_augmented[name_in_dataset]})

        # Add now the measurements that actually need some kind of processing
        final_measurement.update({'speed_module': speed / g_conf.SPEED_FACTOR})
        final_measurement.update({'directions': directions})
        final_measurement.update({'game_time': time_stamp})

        return final_measurement

    def _pre_load_image_folders(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now.

        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """

        episodes_list = glob.glob(os.path.join(path, 'episode_*'))
        sort_nicely(episodes_list)
        # Do a check if the episodes list is empty
        if len(episodes_list) == 0:
            raise ValueError("There are no episodes on the training dataset folder %s" % path)

        sensor_data_names = []
        float_dicts = []

        number_of_hours_pre_loaded = 0

        # Get the suffix of the images
        all_image_name = glob.glob(os.path.join(episodes_list[0], '*RGB_*'))
        all_suffix = set([name.split('.')[-1] for name in all_image_name])
        assert len(all_suffix) == 1, 'The image suffixes should be the same.'
        if 'png' in all_suffix:
            image_suffix = '.png'
        elif 'jpg' in all_suffix:
            image_suffix = '.jpg'
        else:
            raise ValueError("The image suffix should be png or jpg.")

        # Now we do a check to try to find all the
        for episode in episodes_list:

            print('Episode ', episode)

            available_measurements_dict = data_parser.check_available_measurements(episode)

            if number_of_hours_pre_loaded > g_conf.NUMBER_OF_HOURS:
                # The number of wanted hours achieved
                break

            # Get all the measurements from this episode
            measurements_list = glob.glob(os.path.join(episode, 'measurement*'))
            sort_nicely(measurements_list)

            if len(measurements_list) == 0:
                print("EMPTY EPISODE")
                continue

            # A simple count to keep track how many measurements were added this episode.
            count_added_measurements = 0

            for measurement in measurements_list[:-3]:

                data_point_number = measurement.split('_')[-1].split('.')[0]

                with open(measurement) as f:
                    measurement_data = json.load(f)

                # depending on the configuration file, we eliminated the kind of measurements
                # that are not going to be used for this experiment
                # We extract the interesting subset from the measurement dict

                speed = data_parser.get_speed(measurement_data)

                directions = measurement_data['directions']
                if g_conf.DATA_USED != 'side':
                    final_measurement = self._get_final_measurement(speed, measurement_data, 0,
                                                                    directions,
                                                                    available_measurements_dict)

                    if self.is_measurement_partof_experiment(final_measurement):
                        float_dicts.append(final_measurement)
                        rgb = 'CentralRGB_' + data_point_number + image_suffix
                        sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                        count_added_measurements += 1

                # We do measurements for the left side camera
                # We convert the speed to KM/h for the augmentation

                # We extract the interesting subset from the measurement dict

                if g_conf.DATA_USED != 'central':
                    final_measurement = self._get_final_measurement(speed, measurement_data, -30.0,
                                                                    directions,
                                                                    available_measurements_dict)

                    if self.is_measurement_partof_experiment(final_measurement):
                        float_dicts.append(final_measurement)
                        rgb = 'LeftRGB_' + data_point_number + image_suffix
                        sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                        count_added_measurements += 1

                # We do measurements augmentation for the right side cameras

                if g_conf.DATA_USED != 'central':
                    final_measurement = self._get_final_measurement(speed, measurement_data, 30.0,
                                                                    directions,
                                                                    available_measurements_dict)

                    if self.is_measurement_partof_experiment(final_measurement):
                        float_dicts.append(final_measurement)
                        rgb = 'RightRGB_' + data_point_number + image_suffix
                        sensor_data_names.append(os.path.join(episode.split('/')[-1], rgb))
                        count_added_measurements += 1

            # Check how many hours were actually added

            last_data_point_number = measurements_list[-4].split('_')[-1].split('.')[0]
            number_of_hours_pre_loaded += (float(count_added_measurements / 10.0) / 3600.0)
            print(" Loaded ", number_of_hours_pre_loaded, " hours of data")


        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

    def augment_directions(self, directions):

        if directions == 2.0:
            if random.randint(0, 100) < 20:
                directions = random.choice([3.0, 4.0, 5.0])

        return directions

    def augment_steering(self, camera_angle, steer, speed):
        """
            Apply the steering physical equation to augment for the lateral cameras steering
        Args:
            camera_angle: the angle of the camera
            steer: the central steering
            speed: the speed that the car is going

        Returns:
            the augmented steering

        """
        time_use = 1.0
        car_length = 6.0

        pos = camera_angle > 0.0
        neg = camera_angle <= 0.0
        # You should use the absolute value of speed
        speed = math.fabs(speed)
        rad_camera_angle = math.radians(math.fabs(camera_angle))
        val = g_conf.AUGMENT_LATERAL_STEERINGS * (
            math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
        steer -= pos * min(val, 0.3)
        steer += neg * min(val, 0.3)

        steer = min(1.0, max(-1.0, steer))

        return steer

    def augment_measurement(self, measurements, angle, speed, steer_name='steer'):
        """
            Augment the steering of a measurement dict

        """
        new_steer = self.augment_steering(angle, measurements[steer_name],
                                          speed)
        measurements[steer_name] = new_steer
        return measurements

    def controls_position(self):
        return np.where(self.meta_data[:, 0] == b'control')[0][0]


    """
        Methods to interact with the dataset attributes that are used for training.
    """

    def extract_targets(self, data):
        """
        Method used to get to know which positions from the dataset are the targets
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        targets_vec = []
        for target_name in g_conf.TARGETS:
            targets_vec.append(data[target_name])

        return torch.cat(targets_vec, 1)

    def extract_inputs(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        if len(g_conf.INPUTS) != 0:
            for input_name in g_conf.INPUTS:
                inputs_vec.append(data[input_name])
            return torch.cat(inputs_vec, 1)
        else:
            inputs_vec.append(data['speed_module'])
            return torch.cat(inputs_vec, 1)

    def extract_intentions(self, data):
        """
        Method used to get to know which positions from the dataset are the inputs
        for this experiments
        Args:
            labels: the set of all float data got from the dataset

        Returns:
            the float data that is actually targets

        Raises
            value error when the configuration set targets that didn't exist in metadata
        """
        inputs_vec = []
        for input_name in g_conf.INTENTIONS:
            inputs_vec.append(data[input_name])

        return torch.cat(inputs_vec, 1)

    def test(self, img_list, index):

        if not os.path.exists('./processed_image'):
            os.mkdir('./processed_image')
        if len(os.listdir('./processed_image')) == 0:
            frame_idx = 0
        else:
            frame_list = os.listdir('./processed_image')
            frame_list = [int(i.split('_')[-1]) for i in frame_list]
            frame_list.sort()
            frame_idx = frame_list[-1] + 1
        os.mkdir('./processed_image/frame_{}'.format(frame_idx))
        for idx in range(len(img_list)):
            img = img_list[idx]
            img = img.transpose(1, 2, 0)
            cv2.imwrite('./processed_image/frame_{}/{}.png'.format(frame_idx, idx), img)

        img_path = os.path.join(self.root_dir,
                                self.sensor_data_names[index].split('/')[-2],
                                self.sensor_data_names[index].split('/')[-1])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
