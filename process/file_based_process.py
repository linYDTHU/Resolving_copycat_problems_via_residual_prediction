import os
import sys
import scipy
import numpy as np
import glob
import json
import random
from PIL import Image
import cv2
import copy
import argparse

sys.path.append('../')

import carla08 as carla


VIEW_WIDTH = 800
VIEW_HEIGHT = 600
VIEW_FOV = 100

BB_COLOR = (248, 64, 24)

IMAGE_CUT = [90, 485]

# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================


class FileBasedProcess(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """
    def __init__(self, process_method):
        if 'blackhole' in process_method:
            self.process_func = self.process_blackhole
            self.mask_color = 0
            self.model = None
            if 'randombox' in process_method:
                self.randombox = True
            else:
                self.randombox = False
            if 'manual1' in process_method or 'manual_' in process_method:
                self.manual_filter = 1
            elif 'manual2' in process_method:
                self.manual_filter = 2
            else:
                self.manual_filter = 0
        elif 'allblack' in process_method:
            self.process_func = self.process_allblack
            self.mask_color = None
            self.model = None
            self.randombox = False
            self.manual_filter = None
        else:
            self.process_func = self.process_none
            self.mask_color = None
            self.model = None
            self.randombox = False
            self.manual_filter = None

    def process_none(self, image, measurements, mask_probability=1):
        return image

    def process_allblack(self, image, measurements, mask_probability=1):
        return np.zeros_like(image)

    def process_blackhole(self, image, measurements, mask_probability=1):
        if self.manual_filter == 1:
            manually_filter = manually_filter1
        elif self.manual_filter == 2:
            manually_filter = manually_filter2
        else:
            manually_filter = None

        mask = np.ones_like(image) * 255

        height = image.shape[0]
        width = image.shape[1]

        playerAgent = Agent(measurements['playerMeasurements'])

        camera_player_transform = {'transform': {'location': {'x': 2.0, 'y': 0.0, 'z': 1.4},
                                                 'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                                 'rotation': {"pitch": -15.0, "roll": 0.0, "yaw": 0.0}}}
        camera_agent = Agent(camera_player_transform)
        camera_agent = set_calibration(camera_agent, height, width)

        nonPlayerAgents = []
        for item in measurements['nonPlayerAgents']:
            if 'vehicle' in item:
                nonPlayerAgent = Agent(item['vehicle'])
                if manually_filter is not None:
                    matched = manually_filter(nonPlayerAgent, camera_agent, playerAgent, 'vehicle')
                    if matched:
                        nonPlayerAgents.append(nonPlayerAgent)
                else:
                    nonPlayerAgents.append(nonPlayerAgent)
            elif 'pedestrian' in item:
                nonPlayerAgent = Agent(item['pedestrian'])
                if manually_filter is not None:
                    matched = manually_filter(nonPlayerAgent, camera_agent, playerAgent, 'pedestrian')
                    if matched:
                        nonPlayerAgents.append(nonPlayerAgent)
                else:
                    nonPlayerAgents.append(nonPlayerAgent)
            elif 'trafficLight' in item:
                item['trafficLight']['boundingBox'] = {'extent': {"x": 0.3, "y": 0.3, "z": 0.8},
                                                       'transform': {"location": {"y": -0.2, "z": 2.85},
                                                                     "rotation": {}, "orientation": {}}}
                traffic_light_agent = Agent(item['trafficLight'])
                if manually_filter is not None:
                    matched = manually_filter(traffic_light_agent, camera_agent, playerAgent, 'trafficLight')
                    if matched:
                        nonPlayerAgents.append(traffic_light_agent)
                else:
                    nonPlayerAgents.append(traffic_light_agent)

        nonPlayerAgents = filte_out_near_nonplayer(nonPlayerAgents, playerAgent, 50)

        bounding_boxes = FileBasedProcess.get_bounding_boxes(nonPlayerAgents, camera_agent, playerAgent)
        masked_image, mask, _, mask_or_not = FileBasedProcess.draw_mask_image(image, mask, bounding_boxes,
                                                                              probability=mask_probability,
                                                                              mask_color=0)
        if self.randombox:
            for i in range(int(abs(random.gauss(1, 3)))):
                x1 = random.randint(0, masked_image.shape[0] - 2)
                x2 = min(x1 + random.randint(5, 150), masked_image.shape[0])
                y1 = random.randint(0, masked_image.shape[1] - 2)
                y2 = min(y1 + random.randint(5, 150), masked_image.shape[1])
                masked_image[x1:x2, y1:y2, :] = 0.0
        return masked_image

    @staticmethod
    def get_bounding_boxes(vehicles, camera, player):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [FileBasedProcess.get_bounding_box(vehicle, camera, player)[0] for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_mask_image(image, mask, bounding_boxes, probability=1, mask_color=0):
        """
        Draws masks on the image.
        """

        rectangles = []
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            rectangles.append(FileBasedProcess.external_rectangle(points))
        rois = [(rectangle[0][0], rectangle[0][1], rectangle[2][0], rectangle[2][1]) for rectangle in rectangles]
        digged_image, mask_image, avalibale_bbox, mask_or_not = FileBasedProcess.dig_bbox(image, mask, rois, probability, mask_color)
        return digged_image, mask_image, avalibale_bbox, mask_or_not

    @staticmethod
    def get_bbox_annotation(image, bounding_boxes):
        """
        Get the coordinates of the available bounding boxes
        """

        h = image.shape[0]
        w = image.shape[1]

        regions = {}
        index = 0
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            rectangle = FileBasedProcess.external_rectangle(points)
            roi = (rectangle[0][0], rectangle[0][1], rectangle[2][0], rectangle[2][1])
            if ((0 <= roi[1] < h) and (0 <= roi[0] < w)) or\
                    ((0 <= roi[3] < h) and (0 <= roi[2] < w)) or\
                    ((0 <= roi[1] < h) and (0 <= roi[2] < w)) or\
                    ((0 <= roi[3] < h) and (0 <= roi[0] < w)):
                roi = (max(0, roi[0]), max(0, roi[1]), max(0, roi[2]), max(0, roi[3]))
                roi = (int(min(w, roi[0])), int(min(h, roi[1])), int(min(w, roi[2])), int(min(h, roi[3])))
                regions[str(index)] = {'roi': roi, 'name': 'bbox'}
                index += 1
        return regions

    @staticmethod
    def dig_bbox(image, mask_image, rois, probability=1, mask_color=0):
        """
        :param mask_image: numpy.array
        :param image: numpy.array
        :param rois: 2d list, [[w1, h1, w2, h2], ...]
        :param mask_color: int, color of mask, 0/255
        :param output_mask: bool
        :return: numpy.array or (numpy.array, numpy.array)
        """

        if mask_color not in [0, 255]:
            raise NotImplementedError('Only support 0 and 255')

        target_objects = {'person', 'car', 'traffic light', 'bus'}
        # copy the orignal image
        digged_image = copy.deepcopy(image)
        h = digged_image.shape[0]
        w = digged_image.shape[1]
        mask_or_not = False
        avalible_bbox = []
        for i in range(len(rois)):
            bbox = rois[i]
            if ((0 <= bbox[1] < h) and (0 <= bbox[0] < w)) or\
                    ((0 <= bbox[3] < h) and (0 <= bbox[2] < w)) or\
                    ((0 <= bbox[1] < h) and (0 <= bbox[2] < w)) or\
                    ((0 <= bbox[3] < h) and (0 <= bbox[0] < w)):
                bbox = (max(0, bbox[0]), max(0, bbox[1]), max(0, bbox[2]), max(0, bbox[3]))
                bbox = (int(min(w, bbox[0])), int(min(h, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3])))
                if random.random() < probability:
                    digged_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = mask_color
                    mask_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = mask_color
                    mask_or_not = True
                    avalible_bbox.append(bbox)
        return digged_image, mask_image, avalible_bbox, mask_or_not

    @staticmethod
    def draw_bounding_boxes(image, bounding_boxes, bbox_type):
        """
        Draws bounding boxes on the input image.
        """
        if bbox_type == '2D':
            image = FileBasedProcess.draw_2D_bounding_boxes(image, bounding_boxes)
        elif bbox_type == '3D':
            image = FileBasedProcess.draw_3D_bounding_boxes(image, bounding_boxes)
        else:
            raise NotImplementedError
        return image

    @staticmethod
    def external_rectangle(points):
        coord_array = np.array(points)
        x_min = np.min(coord_array[:, 0])
        x_max = np.max(coord_array[:, 0])
        y_min = np.min(coord_array[:, 1])
        y_max = np.max(coord_array[:, 1])
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    @staticmethod
    def draw_2D_bounding_boxes(image, bounding_boxes):
        """
        Draws 2D bounding boxes on the input image.
        """

        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            rectangle = FileBasedProcess.external_rectangle(points)
            cv2.line(image, rectangle[0], rectangle[1], color=BB_COLOR, thickness=2)
            cv2.line(image, rectangle[1], rectangle[2], color=BB_COLOR, thickness=2)
            cv2.line(image, rectangle[2], rectangle[3], color=BB_COLOR, thickness=2)
            cv2.line(image, rectangle[3], rectangle[0], color=BB_COLOR, thickness=2)
        return image

    @staticmethod
    def draw_3D_bounding_boxes(image, bounding_boxes):
        """
        Draws 3D bounding boxes on the input image.
        """

        # bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        # bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            cv2.line(image, points[0], points[1], color=BB_COLOR, thickness=2)
            cv2.line(image, points[1], points[2], color=BB_COLOR, thickness=2)
            cv2.line(image, points[2], points[3], color=BB_COLOR, thickness=2)
            cv2.line(image, points[3], points[0], color=BB_COLOR, thickness=2)
            # pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            # pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            # pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            # pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            # pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            cv2.line(image, points[4], points[5], color=BB_COLOR, thickness=2)
            cv2.line(image, points[5], points[6], color=BB_COLOR, thickness=2)
            cv2.line(image, points[6], points[7], color=BB_COLOR, thickness=2)
            cv2.line(image, points[7], points[4], color=BB_COLOR, thickness=2)
            # pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            # pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            # pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            # pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            cv2.line(image, points[0], points[4], color=BB_COLOR, thickness=2)
            cv2.line(image, points[1], points[5], color=BB_COLOR, thickness=2)
            cv2.line(image, points[2], points[6], color=BB_COLOR, thickness=2)
            cv2.line(image, points[3], points[7], color=BB_COLOR, thickness=2)
        #     pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
        #     pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
        #     pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
        #     pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        # display.blit(bb_surface, (0, 0))
        return image

    @staticmethod
    def get_bounding_box(vehicle, camera, player):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = FileBasedProcess._create_bb_points(vehicle)
        cords_x_y_z = FileBasedProcess._vehicle_to_sensor(bb_cords, vehicle, camera, player)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox, cords_x_y_z

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.boundingBox.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor, player):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = FileBasedProcess._vehicle_to_world(cords, vehicle)
        sensor_cord = FileBasedProcess._world_to_sensor(world_cord, sensor, player)
        return sensor_cord

    @staticmethod
    def _complete_transform(transform):
        """
        Complete the missing items in transform
        """

        if 'x' not in transform.location.attr:
            transform.location.x = 0.0
        if 'y' not in transform.location.attr:
            transform.location.y = 0.0
        if 'z' not in transform.location.attr:
            transform.location.z = 0.0
        if 'yaw' not in transform.rotation.attr:
            transform.rotation.yaw = 0.0
        if 'roll' not in transform.rotation.attr:
            transform.rotation.roll = 0.0
        if 'pitch' not in transform.rotation.attr:
            transform.rotation.pitch = 0.0
        return transform

    @staticmethod
    def _complete_bb_transform(vehicle):
        """
        Complete the missing items in transform
        """

        if 'x' not in vehicle.boundingBox.transform.location.attr:
            vehicle.boundingBox.transform.location.x = vehicle.transform.location.x
        else:
            vehicle.boundingBox.transform.location.x += vehicle.transform.location.x
        if 'y' not in vehicle.boundingBox.transform.location.attr:
            vehicle.boundingBox.transform.location.y = vehicle.transform.location.y
        else:
            vehicle.boundingBox.transform.location.y += vehicle.transform.location.y
        if 'z' not in vehicle.boundingBox.transform.location.attr:
            vehicle.boundingBox.transform.location.z = vehicle.transform.location.z
        else:
            vehicle.boundingBox.transform.location.z += vehicle.transform.location.z
        if 'yaw' not in vehicle.boundingBox.transform.rotation.attr:
            vehicle.boundingBox.transform.rotation.yaw = vehicle.transform.rotation.yaw
        else:
            vehicle.boundingBox.transform.rotation.yaw += vehicle.transform.rotation.yaw
        if 'roll' not in vehicle.boundingBox.transform.rotation.attr:
            vehicle.boundingBox.transform.rotation.roll = vehicle.transform.rotation.roll
        else:
            vehicle.boundingBox.transform.rotation.roll += vehicle.transform.rotation.roll
        if 'pitch' not in vehicle.boundingBox.transform.rotation.attr:
            vehicle.boundingBox.transform.rotation.pitch = vehicle.transform.rotation.pitch
        else:
            vehicle.boundingBox.transform.rotation.pitch += vehicle.transform.rotation.pitch
        return vehicle.boundingBox.transform

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = FileBasedProcess._complete_transform(vehicle.boundingBox.transform)
        bb_vehicle_matrix = FileBasedProcess.get_matrix(bb_transform)
        vehicle_world_matrix = FileBasedProcess.get_matrix(FileBasedProcess._complete_transform(vehicle.get_transform()))
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor, player):
        """
        Transforms world coordinates to sensor.
        """

        sensor_player_matrix = FileBasedProcess.get_matrix(sensor.get_transform())
        player_world_matrix = FileBasedProcess.get_matrix(FileBasedProcess._complete_transform(player.get_transform()))
        sensor_world_matrix = np.dot(player_world_matrix, sensor_player_matrix)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


class Agent(object):
    def __init__(self, measure):
        super(Agent, self).__init__()
        self.attr = {}
        for k, v in measure.items():
            if isinstance(v, dict):
                self.attr[k] = Agent(v)
            else:
                self.attr[k] = v

    def __getattr__(self, item):
        return self.attr[str(item)]

    def get_transform(self):
        if 'transform' in self.attr:
            return self.attr['transform']
        else:
            raise Exception


def set_calibration(camera, height, width):
    calibration = np.identity(3)
    calibration[0, 2] = width / 2.0
    calibration[1, 2] = height / 2.0
    calibration[0, 0] = width / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    calibration[1, 1] = width / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    camera.calibration = calibration
    return camera


def filte_out_near_nonplayer(nonPlayerAgents, playerAgent, threshold=50):
    player_location = np.array([playerAgent.transform.location.x, playerAgent.transform.location.y, playerAgent.transform.location.z])
    near_nonPlayerAgents = []
    for nonplayer in nonPlayerAgents:
        nonplayer_location = np.array([nonplayer.transform.location.x, nonplayer.transform.location.y, nonplayer.transform.location.z])
        dis = np.linalg.norm(player_location - nonplayer_location)
        if dis <= threshold:
            near_nonPlayerAgents.append(nonplayer)
    return near_nonPlayerAgents


def manually_filter1(nonPlayerAgent, cameraAgent, playerAgent, nonPlayerAgent_type, dist_threshold=50):
    """
    :return: True: mask out this agent; False: don't mask this agent
    """
    player_location = np.array([playerAgent.transform.location.x, playerAgent.transform.location.y, playerAgent.transform.location.z])
    nonplayer_location = np.array([nonPlayerAgent.transform.location.x, nonPlayerAgent.transform.location.y, nonPlayerAgent.transform.location.z])
    dis = np.linalg.norm(player_location - nonplayer_location)
    if dis > dist_threshold:
        return False
    _, cord_x_y_z = FileBasedProcess.get_bounding_box(nonPlayerAgent, cameraAgent, playerAgent)
    if (cord_x_y_z[0, :] > 0).all():  # filter out all the nonPlayerAgent behind the Player
        if nonPlayerAgent_type == 'vehicle':
            if not (cord_x_y_z[1, :] < 0).all():  # mask out all the vehicles in front of the Player
                return True
            else:
                return False
        elif nonPlayerAgent_type == 'pedestrian':  # don't mask any pedestraine
            return False
        elif nonPlayerAgent_type == 'trafficLight':  # mask out all the trafficLight in the right hand
            if (cord_x_y_z[1, :] > 0).all():
                return True
            else:
                return False
    else:
        return False


def manually_filter2(nonPlayerAgent, cameraAgent, playerAgent, nonPlayerAgent_type, dist_threshold=50):
    """
    :return: True: mask out this agent; False: don't mask this agent
    """
    player_location = np.array([playerAgent.transform.location.x, playerAgent.transform.location.y, playerAgent.transform.location.z])
    nonplayer_location = np.array([nonPlayerAgent.transform.location.x, nonPlayerAgent.transform.location.y, nonPlayerAgent.transform.location.z])
    dis = np.linalg.norm(player_location - nonplayer_location)
    if dis > dist_threshold:
        return False
    _, cord_x_y_z = FileBasedProcess.get_bounding_box(nonPlayerAgent, cameraAgent, playerAgent)
    if (cord_x_y_z[0, :] > 0).all():  # filter out all the nonPlayerAgent behind the Player
        if nonPlayerAgent_type == 'vehicle':
            if not (cord_x_y_z[1, :] < 0).all():  # mask out all the side vehicles
                return False
            else:
                return True
        elif nonPlayerAgent_type == 'pedestrian':  # mask all the pedestrians
            return True
        elif nonPlayerAgent_type == 'trafficLight':  # mask out all the trafficLight in the left hand
            if (cord_x_y_z[1, :] > 0).all():
                return False
            else:
                return True
    else:
        return False


def reconstruct_image(path, h1, h2):
    """
    Because the original image is cut and resized from the camera data,
    to get accurate 2D coordinates we should reconstruct the original size.
    :param path: the image path
    :param h1: the cut index1
    :param h2: the cut index2
    :return: the image in size (600, 800, 3) in my case
    """
    image = cv2.imread(path)
    image = cv2.resize(image, (VIEW_WIDTH, h2 - h1))
    image_padd = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3))
    image_padd[h1:h2, ...] = image
    return image_padd


def prepare_data_for_coiltraine(args):
    img_root_dir = args.img_dir
    if args.episodes is not None:
        episode_list = args.episodes
        episode_list = ['episode_{:05d}'.format(idx) for idx in episode_list]
    else:
        episode_list = os.listdir(img_root_dir)
        episode_list.remove('metadata.json')

    if args.target_dir is None:
        if args.mask_color == 255:
            target_dir = img_root_dir + '_whitehole'
        elif args.mask_color == 0:
            target_dir = img_root_dir + '_blackhole'
        else:
            raise NotImplementedError
    else:
        target_dir = args.target_dir

    if args.mask_probability != 1.0:
        target_dir = target_dir + '_p{}'.format(str(args.mask_probability).replace('.', ''))

    mask_target_dir = target_dir + '_mask'

    if args.random_box:
        if not 'randombox' in target_dir:
            target_dir = target_dir + '+randombox'

    for episode in episode_list:
        img_path_list = glob.glob(os.path.join(img_root_dir, episode, '*.png'))
        for img_path in img_path_list:
            measurement_path = os.path.join(img_root_dir, episode, 'measurements_'+img_path.split('_')[-1].split('.')[0]+'.json')

            image = reconstruct_image(img_path, IMAGE_CUT[0], IMAGE_CUT[1])

            if args.mask_color == 255:
                mask = np.zeros_like(image)
            elif args.mask_color == 0:
                mask = np.ones_like(image) * 255
            else:
                raise NotImplementedError('Only support 0 and 255')

            with open(measurement_path, 'r') as f:
                measurement = json.load(f)

            nonPlayerAgents = []
            for item in measurement['nonPlayerAgents']:
                if 'vehicle' in item:
                    nonPlayerAgents.append(Agent(item['vehicle']))
                elif 'pedestrian' in item:
                    nonPlayerAgents.append(Agent(item['pedestrian']))
                elif 'trafficLight' in item:
                    item['trafficLight']['boundingBox'] = {'extent': {"x": 0.2, "y": 0.2, "z": 0.7},
                                                           'transform': {"location": {"y": -0.2, "z": 2.9},
                                                                         "rotation": {}, "orientation": {}}}
                    traffic_light_agent = Agent(item['trafficLight'])
                    nonPlayerAgents.append(traffic_light_agent)

            playerAgent = Agent(measurement['playerMeasurements'])

            nonPlayerAgents = filte_out_near_nonplayer(nonPlayerAgents, playerAgent, 50)

            if 'CentralRGB' in img_path:
                yaw = 0.0
            elif 'LeftRGB' in img_path:
                yaw = -30.0
            else:
                yaw = 30.0

            camera_player_transform = {'transform': {'location': {'x': 2.0, 'y': 0.0, 'z': 1.4},
                                                     'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                                                     'rotation': {"pitch": -15.0, "roll": 0.0, "yaw": yaw}}}
            camera_agent = Agent(camera_player_transform)
            camera_agent = set_calibration(camera_agent)

            bounding_boxes = FileBasedProcess.get_bounding_boxes(nonPlayerAgents, camera_agent, playerAgent)
            masked_image, mask, _, mask_or_not = FileBasedProcess.draw_mask_image(image, mask, bounding_boxes,
                                                                                         probability=args.mask_probability,
                                                                                         mask_color=args.mask_color)

            if args.random_box:
                for i in range(int(abs(random.gauss(1, 3)))):
                    x1 = random.randint(0, masked_image.shape[0]-2)
                    x2 = min(x1 + random.randint(5, 150), masked_image.shape[0])
                    y1 = random.randint(0, masked_image.shape[1]-2)
                    y2 = min(y1 + random.randint(5, 150), masked_image.shape[1])
                    masked_image[x1:x2, y1:y2, :] = args.mask_color

            if args.mask_color == 0:
                masked_image = masked_image[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
                masked_image = cv2.resize(masked_image, (200, 88),  interpolation=cv2.INTER_AREA)
                mask = mask[IMAGE_CUT[0]:IMAGE_CUT[1], ...]
                mask = cv2.resize(mask, (200, 88),  interpolation=cv2.INTER_AREA)

            if not os.path.exists(os.path.join(target_dir, episode)):
                os.makedirs(os.path.join(target_dir, episode))
            print(os.path.join(target_dir, episode, img_path.split('/')[-1]))
            cv2.imwrite(os.path.join(target_dir, episode, img_path.split('/')[-1]), masked_image)

            if args.output_mask:
                if not os.path.exists(os.path.join(mask_target_dir, episode)):
                    os.makedirs(os.path.join(mask_target_dir, episode))
                cv2.imwrite(os.path.join(mask_target_dir, episode, img_path.split('/')[-1]), mask)

