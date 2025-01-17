#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:48:46 2023

@author: oscar
"""
import sys
# sys.path.append('home/oscar/Dropbox/InterFuser')
sys.path.append('home/automan/Dropbox/InterFuser')

import os
import json
import time
import datetime
import pathlib
import time
import imp
import cv2
from collections import deque
import matplotlib.pyplot as plt

import torch
import numpy as np
from PIL import Image
from easydict import EasyDict

import carla
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from timm.models import create_model
from team_code.map_agent import MapAgent
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker

from corl_config import GlobalConfig
# from carla_evaluate.corl_controller import PIDController, CORLController
from team_code.pid_controller import PIDController
from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.model.generate_target_coords import inference_once


import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

WEATHERS = {
    "ClearNoon": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, 45.0, 75.0, 0.0, 0.0, 0.0, 0.0),
    "ClearSunset": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, 45.0, 5.0, 0.0, 0.0, 0.0, 0.0),
    "ClearNight": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, -1.0, -90.0, 0.0, 0.0, 0.0, 0.0),
    # "ClearNight": carla.WeatherParameters(5.0, 0.0, 0.0, 0.35, -1.0, -90.0, 60.0, 75.0, 1.0, 0.0),
    # "CloudyMorning": carla.WeatherParameters(60.0, 0.0, 30.0, 10.0, 45.0, 35.0, 0.0, 0.0, 0.0, 0.0),
    # "CloudyDawn": carla.WeatherParameters(60.0, 0.0, 30.0, 10.0, 45.0, 5.0, 0.0, 0.0, 0.0, 0.0),
    "FogyNoon": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, 45.0, 75.0, 40.0, 0.75, 1.0, 0.0),
    "FogyySunset": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, 45.0, 5.0, 20.0, 0.75, 1.0, 0.0),
    "FogyNight": carla.WeatherParameters(60.0, 0.0, 50.0, 10.0, -1.0, -90.0, 60.0, 0.75, 1.0, 0.0),
    "HardRainNoon": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, 45.0, 75.0, 0.0, 0.0, 0.0, 100.0),
    "HardRainSunset": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, 45.0, 5.0, 0.0, 0.0, 0.0, 100.0),
    # "HardRainNight": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, -1.0, -90.0, 100.0, 0.75, 0.1, 100.0),
    "HardRainNight": carla.WeatherParameters(100.0, 100.0, 70.0, 50.0, -1.0, -90.0, 60.0, 30.0, 1.0, 100.0),
    # "ClearNoon": carla.WeatherParameters.ClearNoon,
    # "ClearSunset": carla.WeatherParameters.ClearSunset,
    #"CloudyNoon": carla.WeatherParameters.CloudyNoon,
    #"CloudySunset": carla.WeatherParameters.CloudySunset,
    #"WetNoon": carla.WeatherParameters.WetNoon,
    #"WetSunset": carla.WeatherParameters.WetSunset,
    #"WetNight": carla.WeatherParameters(5.0,0.0,50.0,10.0,-1.0,-90.0,60.0,75.0,1.0,60.0),
    #"WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    #"WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    #"WetCloudyNight": carla.WeatherParameters(60.0,0.0,50.0,10.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    #"SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    #"SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    #"SoftRainNight": carla.WeatherParameters(60.0,30.0,50.0,30.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    # "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    # "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    # "MidRainyNight": carla.WeatherParameters(80.0,60.0,60.0,60.0,-1.0,-90.0,60.0,0.75,0.1,80.0),
    # "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    # "HardRainSunset": carla.WeatherParameters.HardRainSunset,
}
WEATHERS_IDS = list(WEATHERS)

DECISION_LIST = ['Left', 'Right', 'STRAIGHT', 'Keep Lane', 'ChangeLane Left', 'ChangeLane Right']

class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        pygame.display.set_caption("CORL Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        # map = input_data['map']
        trajectory = input_data['predicted_trajectory']
        # lat_decision = input_data['predicted_lat_decision']
        # long_decision = input_data['predicted_long_decision']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = input_data['bev']
        surface[440:600,1000:1200] = trajectory[0:160,:]
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        # surface = cv2.putText(surface, input_data['speed'], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Behavior Decision', (820,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 'Planned Trajectory', (1010,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        # surface = cv2.putText(surface, long_decision, (820,480), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)
        # surface = cv2.putText(surface, lat_decision, (820,530), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255

        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()

def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    return collides, p1 + x[0] * v1

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result

def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
        )
        b = np.sqrt(
            vehicle.bounding_box.extent.x**2
            + vehicle.bounding_box.extent.y**2
            + vehicle.bounding_box.extent.z**2
        )

        if dist > a + b:
            continue

        result.append(light)

    return result

def get_entry_point():
    return "InterfuserAgent"

class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img

def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InterfuserAgent(autonomous_agent.AutonomousAgent):
# class InterfuserAgent(MapAgent):
    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters
    
    def setup(self, path_to_conf_file):

        # self._hic = DisplayInterface()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('use:', self.device)
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.control_step = 0 #-1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        # self.rgb_front_transform = create_carla_rgb_transform(336)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()
        
        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig(setting='eval')
        self.skip_frames = self.config.skip_frames
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len

        # load LLM model and text decoder settings
        tokenizer, model, image_processor, context_len = load_pretrained_model(self.config.model_path, False, False)
        self.tokenizer = tokenizer
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.image_processor = image_processor
    
        # self.img_states = np.zeros(shape=(self.seq_len, 3,
        #                                   self.config.img_resolution[0],
        #                                   self.config.img_resolution[1]),
        #                            dtype=np.int32,
        #                            )
        init_img_list = []
        for idx in range(self.seq_len):
            img_temp = self.image_processor(np.zeros((3, self.config.img_resolution[0], self.config.img_resolution[1])), return_tensors='np')['pixel_values'][0]
            init_img_list.append(img_temp)
        self.img_states = np.stack(init_img_list)
        
        self.actions = np.zeros(shape=(0, self.config.act_dim))
        self.rtgs = np.ones(shape=(self.seq_len, 1)) * 30.0 #15.06
        self.decision_states =  None #np.zeros(shape=(0, len(RoadOption) - 1))
        self.detection_states = np.zeros(shape=(0, 4))
        self.timesteps = np.array(([0.0], [1.0], [2.0]))
        self.target_waypoints = np.zeros(shape=(0, 2))
        
        self.rgb = None
        self.action = None
        self.reward = None
        self.timestep = 1
        self.decision = None
        self.detection = None

        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_action = np.zeros(2)
        self.prev_surround_map = None
        self.reward_scale = 100

        control = carla.VehicleControl()
        control.steer = float(0.0)
        control.throttle = float(0.0)
        control.brake = float(0.0)

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):        
        self._route_planner = RoutePlanner(7.5, 25.0, 257)
        self._route_planner.set_route(self._global_plan, True)
        
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        print(len(self._waypoint_planner.route))
        
        # self.controller = CORLController(self.config)
        self._turn_controller = PIDController(K_P=1.4, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        
        self.initialized = True
        self._hic = DisplayInterface()
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        
        if self.config.weather != "none":
            weather = WEATHERS[self.config.weather]
            self._world.set_weather(weather)
        
        self._map = self._world.get_map()
        # self._actors = self._world.get_actors()

        # self._traffic_lights = get_nearby_lights(
        #     self._vehicle, self._actors.filter("*traffic_light*")
        # )
        
        # lights_list = self._world.get_actors().filter("*traffic_light*")
        # self._list_traffic_lights = []
        # for light in lights_list:
        #     center, waypoints = self.get_traffic_light_waypoints(light)
        #     self._list_traffic_lights.append((light, center, waypoints))
        # (
        #     self._list_traffic_waypoints,
        #     self._dict_traffic_lights,
        # ) = self._gen_traffic_light_dict(self._list_traffic_lights)
    
        self.birdview_producer = BirdViewProducer(
            CarlaDataProvider.get_client(),  # carla.Client
            target_size=PixelDimensions(width=400, height=400),
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )
        
        # for stop signs
        self._target_stop_sign = None  # the stop sign affecting the ego vehicle
        self._stop_completed = False  # if the ego vehicle has completed the stop sign
        self._affected_by_stop = (
            False  # if the ego vehicle is influenced by a stop sign
        )
    
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def get_traffic_light_waypoints(self, traffic_light):
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(
            -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
        )  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if (
                not ini_wps
                or ini_wps[-1].road_id != wpx.road_id
                or ini_wps[-1].lane_id != wpx.lane_id
            ):
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)

    def _gen_traffic_light_dict(self, traffic_lights_list):
        traffic_light_dict = {}
        waypoints_list = []
        for light, center, waypoints in traffic_lights_list:
            for waypoint in waypoints:
                traffic_light_dict[waypoint] = (light, center)
                waypoints_list.append(waypoint)
        return waypoints_list, traffic_light_dict

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = gps #pos
        result["pos"] = pos
        #######################################################################
        next_wp, next_cmd = self._waypoint_planner.run_step(pos)
        target_wp, target_cmd = self._route_planner.run_step(pos)
        #######################################################################
        
        result['next_wp'] = next_wp
        result["next_cmd"] = next_cmd.value
        result['target_wp'] = target_wp
        result['target_cmd'] = target_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([target_wp[0] - pos[0], target_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_local_waypoint"] = local_command_point

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        start_time = time.time()
        if not self.initialized:
            self._init()        

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]
        
        #######################################################################
        command = tick_data["next_cmd"]
        # command = self._command_planner.route[0][1].value
        #######################################################################
        
        rgb = cv2.resize(tick_data["rgb"], tuple(self.config.img_resolution))
        rgb = np.transpose(rgb, (2, 0, 1))
        timestep = np.array(self.step+3)[np.newaxis]
        target_waypoint, target_command = tick_data['target_local_waypoint'], None

        self.states_adapter(rgb, timestep, decision_states=None, target_waypoints=target_waypoint)
        
        # target_waypoints = torch.tensor(self.target_waypoints, dtype=torch.bfloat16).unsqueeze(0).to(device=self.device)
        # convert the waypoint value to text
        
        trajectory = inference_once(self.model, self.tokenizer, self.img_states, self.target_waypoints)
        
        ##### Safe Controller #####
        steer, throttle, brake, target_speed = self.control_pid(trajectory, velocity)
        print('steer', steer)
        print('throttle', throttle)
        print('brake is ', brake)

        control = carla.VehicleControl()
        control.steer = float(steer) * 0.8
        control.throttle = float(throttle)
        control.brake = float(brake)

        # lateral displacement reward
        ego_theta = tick_data["compass"]
        pos = tick_data["pos"]
        ego_x = pos[0]
        ego_y = pos[1]
        loc = self._vehicle.get_location()
        ego_waypoint = self._map.get_waypoint(loc)
        wp_x = - ego_waypoint.transform.location.y
        wp_y = ego_waypoint.transform.location.x
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        ego_waypoint_local = R.T.dot(np.array([wp_x - ego_x, wp_y - ego_y]))

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0], pixels_per_meter=10, max_distance=10,
        )

        render_trajectory = render_waypoints(trajectory, pixels_per_meter=30, max_distance=15, color=(0, 255, 0))

        self_car_map = cv2.resize(self_car_map, (200, 200))
        render_trajectory = cv2.resize(render_trajectory, (200, 200))

        surround_map = np.clip(
            (
                # surround_map.astype(np.float32)
                self_car_map.astype(np.float32)
                + render_trajectory.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)
        tick_data["predicted_trajectory"] = surround_map
        # tick_data["predicted_lat_decision"] = LAT_DECISION_LIST[next_route_command-1]

        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        # tick_data["speed"] = "speed: %.2f Km/h, target_speed: %.2f Km/h" % (velocity*3.6, target_speed*3.6)
        bev = tick_data["raw_lidar"]
        bev = BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=self._vehicle)
        )
        tick_data["bev"] = cv2.resize(bev, (400, 400))
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp
        surface = self._hic.run_interface(tick_data)
        tick_data["surface"] = surface

        end_time = time.time()
        print('infer time is %.1f' % (end_time - start_time))
        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        return

    def destroy(self):
        del self.model

    def actions_rewards_adapter(self, actions, rewards):
        self.actions = np.concatenate((self.actions, actions[np.newaxis]), axis=0)[-self.seq_len:]
        self.rtgs = np.concatenate((self.rtgs, (self.rtgs[-1] - rewards/self.reward_scale)[np.newaxis]), axis=0)[-self.seq_len:]

    def states_adapter(self, img_states, timesteps, decision_states=None, target_waypoints=None):
        # convert to normalized image
        img_states = self.image_processor.preprocess(img_states, return_tensors='np')['pixel_values'][0]
        self.img_states = np.concatenate((self.img_states, img_states[np.newaxis]), axis=0)[-self.seq_len:]
        # self.detection_states = np.concatenate((self.detection_states, detection_states[np.newaxis]), axis=0)[-self.seq_len:]
        self.timesteps = np.concatenate((self.timesteps, timesteps[np.newaxis]), axis=0)[-self.seq_len:]

        if decision_states is not None:
            self.decision_states = np.concatenate((self.decision_states, decision_states[np.newaxis]), axis=0)[-self.seq_len:]

        if target_waypoints is not None:
            self.target_waypoints = np.concatenate((self.target_waypoints, target_waypoints[np.newaxis]), axis=0)[-self.seq_len:]

    def _get_angle_to(self, pos, theta, target):
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def control_pid(self, waypoints, velocity):
        '''
        Borrowed from LMDrive.
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''

        speed = velocity
        
        try:
            # desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

            # could work setting1
            desired_speed = np.linalg.norm(waypoints[4] - waypoints[0]) / 2.0

            # setting2
            # desired_speed = np.linalg.norm(waypoints[3] - waypoints[0]) / 2
        except:
            import pdb; pdb.set_trace()
        
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio
        
        # could work setting1
        aim = (waypoints[2] + waypoints[0]) / 2.0

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self._turn_controller.step(angle)
        print(steer)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0
        
        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata
        """Convert the vehicle transform directly to forward speed"""
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        speed = np.dot(vel_np, orientation)
        return speed