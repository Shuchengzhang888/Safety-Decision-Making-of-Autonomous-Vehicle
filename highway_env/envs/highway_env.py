import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.vehicle.behavior import IDMVehicle

Observation = np.ndarray

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 30,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 120,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 1.8,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.8,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 40],
            "offroad_terminal": False,
            # "show_trajectories": True,
            "screen_width": 800,
            "screen_height": 400,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=40),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        reference = False
        for others in other_per_controlled:
            if not reference:
                vehicle = Vehicle.create_random(
                    self.road,
                    speed=45,
                    spacing=self.config["ego_spacing"],
                )
                MDPvehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
                MDPvehicle.MIN_SPEED = 0
                MDPvehicle.trajectory = vehicle.trajectory
            else:
                MDPvehicle =other_vehicles_type.create_random(
                self.road,
                speed=45,
                spacing=self.config["ego_spacing"]
            )

            self.controlled_vehicles.append(MDPvehicle)
            self.road.vehicles.append(MDPvehicle)

            while(len(self.road.vehicles) != others+1):
                v = other_vehicles_type.create_random(self.road, road_length = 10000, spacing=1 / self.config["vehicles_density"])
                for exsiting_v in self.road.vehicles:
                    if np.linalg.norm(v.position - exsiting_v.position) < 5:
                        break
                else:
                    v.MAX_SPEED -= 20
                    #vehicle.MIN_SPEED += 10
                    v.randomize_behavior()
                    self.road.vehicles.append(v)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        # if (action == 2 and lane == 3) or (action == 0 and lane == 0):
        #     reward -= 5
        if forward_speed <= 1:
            reward -= 1
        # if action == 0 or action == 2:
        #     reward += self.config["lane_change_reward"]
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        ######################
        test_length = 2300
        ######################
        if self.vehicle.position[0] >= test_length:
            return True
        if self.vehicle.speed <= 1:
            return True
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)
    
    def _is_truncated(self) -> bool:
        return False

    def _is_done(self) -> bool:
        return self._is_terminal() or self._is_truncated()

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvCircle(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 12,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 60,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.3,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.8,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [8, 20],
            "offroad_terminal": False,
            "screen_width": 800,
            "screen_height": 400,
            ##"show_trajectories": True,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        net = RoadNetwork()

        speedlimits = [30, 30, 30, 30]

        # Initialise First Lane
        lane = StraightLane([0, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([0, 5], [100, 5], line_types=(LineType.STRIPED, LineType.NONE), width=5, speed_limit=speedlimits[1]))
        net.add_lane("a", "b", StraightLane([0, 10], [100, 10], line_types=(LineType.STRIPED, LineType.NONE), width=5, speed_limit=speedlimits[1]))
        net.add_lane("a", "b", StraightLane([0, 15], [100, 15], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+10, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+15, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -120],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -120],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([130, -20], [130, -120],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([135, -20], [135, -120],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [100, -120]
        radii2 = 20
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(5), np.deg2rad(-95), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+5, np.deg2rad(5), np.deg2rad(-95), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+10, np.deg2rad(5), np.deg2rad(-95), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+15, np.deg2rad(5), np.deg2rad(-95), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 5 Horizontal straight
        net.add_lane("e", "f", StraightLane([100, -140], [0, -140], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[1]))
        net.add_lane("e", "f", StraightLane([100, -145], [0, -145], line_types=(LineType.STRIPED, LineType.NONE), width=5, speed_limit=speedlimits[1]))
        net.add_lane("e", "f", StraightLane([100, -150], [0, -150], line_types=(LineType.STRIPED, LineType.NONE), width=5, speed_limit=speedlimits[1]))
        net.add_lane("e", "f", StraightLane([100, -155], [0, -155], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))

        # 6 - Circular Arc #3
        center3 = [0, -120]
        radii3 = 20
        net.add_lane("f", "g",
                     CircularLane(center3, radii3, np.deg2rad(-85), np.deg2rad(-185), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("f", "g",
                     CircularLane(center3, radii3+5, np.deg2rad(-85), np.deg2rad(-185), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("f", "g",
                     CircularLane(center3, radii3+10, np.deg2rad(-85), np.deg2rad(-185), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("f", "g",
                     CircularLane(center3, radii3+15, np.deg2rad(-85), np.deg2rad(-185), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 7 - Vertical Straight
        net.add_lane("g", "h", StraightLane([-20, -120], [-20, -20],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("g", "h", StraightLane([-25, -120], [-25, -20],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("g", "h", StraightLane([-30, -120], [-30, -20],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("g", "h", StraightLane([-35, -120], [-35, -20],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 8 - Circular Arc #4
        center4 = [0, -20]
        radii4 = 20
        net.add_lane("h", "a",
                     CircularLane(center4, radii4, np.deg2rad(185), np.deg2rad(85), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("h", "a",
                     CircularLane(center4, radii4+5, np.deg2rad(185), np.deg2rad(85), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("h", "a",
                     CircularLane(center4, radii4+10, np.deg2rad(185), np.deg2rad(85), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("h", "a",
                     CircularLane(center4, radii4+15, np.deg2rad(185), np.deg2rad(85), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))


        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        rng = self.np_random
        self.controlled_vehicles = []
        reference = False
        for others in other_per_controlled:
            if not reference:
                vehicle = Vehicle.create_random(
                    self.road,
                    speed=10,
                    lane_from='a',
                    lane_to='b',
                    lane_id=1,
                    spacing=self.config["ego_spacing"],
                )
                MDPvehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
                MDPvehicle.MIN_SPEED = 0
                MDPvehicle.trajectory = vehicle.trajectory
            else:
                MDPvehicle =other_vehicles_type.create_random(
                self.road,
                speed=20,
                spacing=self.config["ego_spacing"]
            )

            self.controlled_vehicles.append(MDPvehicle)
            self.road.vehicles.append(MDPvehicle)

            from highway_env.vehicle.kinematics  import record_trajectory
            while(len(self.road.vehicles) != others+1):
                random_lane_index = self.road.network.random_lane_index(rng)
                v = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                            longitudinal=rng.uniform(low=0, high=self.road.network.get_lane(random_lane_index).length),
                                            speed=10)
                v.id = len(self.road.vehicles)
                v.trajectory.append(record_trajectory(v.road, v.position, v.velocity, v.lane_offset[2], v.heading))
                v.trajectory.append(record_trajectory(v.road, v.position, v.velocity, v.lane_offset[2], v.heading))
                v.trajectory.append(record_trajectory(v.road, v.position, v.velocity, v.lane_offset[2], v.heading))
                dt = 0.25
                #######################
                delta_f = v.action['steering']
                beta = np.arctan(1 / 2 * np.tan(delta_f))

                pred_v1 = v.speed * np.array([np.cos(v.heading + beta), np.sin(v.heading + beta)])
                pred_position1 = v.position + pred_v1*dt
                pred_heading1 = v.heading + v.speed * np.sin(beta) / (v.LENGTH / 2) * dt
                pred_speed1 = v.speed + v.action['acceleration'] * dt
                long, _ = v.lane.local_coordinates(pred_position1)
                pred_angle_offset1 = v.lane.local_angle(pred_heading1, long)
                v.trajectory.append(record_trajectory(v.road, pred_position1, pred_v1, pred_angle_offset1, pred_heading1))

                pred_v2 = pred_speed1 * np.array([np.cos(pred_heading1 + beta), np.sin(pred_heading1 + beta)])
                pred_position2 = pred_position1 + pred_v2*dt
                pred_heading2 = pred_heading1 + pred_speed1 * np.sin(beta) / (v.LENGTH / 2) * dt
                long, _ = v.lane.local_coordinates(pred_position2)
                pred_angle_offset2 = v.lane.local_angle(pred_heading2, long)
                v.trajectory.append(record_trajectory(v.road, pred_position2, pred_v2, pred_angle_offset2, pred_heading2))

                assert len(v.trajectory) == 5
                #################################
                # Prevent early collisions
                for exsiting_v in self.road.vehicles:
                    if np.linalg.norm(v.position - exsiting_v.position) < 20:
                        break
                else:
                    v.MAX_SPEED -= 25
                    #vehicle.MIN_SPEED += 10
                    v.randomize_behavior()
                    self.road.vehicles.append(v)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # print(self.controlled_vehicles[0].position,self.controlled_vehicles[0].new_heading, np.sin(self.controlled_vehicles[0].new_heading),np.cos(self.controlled_vehicles[0].new_heading) )
        # print(self.controlled_vehicles[0].position,self.controlled_vehicles[0].heading, np.sin(self.controlled_vehicles[0].heading),np.cos(self.controlled_vehicles[0].heading) )
        # print(self.controlled_vehicles[0].lane_offset[2])
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        if forward_speed <= 1:
            reward -= 1

        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        # if self.vehicle.position[0] < -20 and self.vehicle.position[1] > -25:
        #     return True
        if self.vehicle.speed <= 1:
            return True
        return self.vehicle.crashed or \
            self.time >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)


class HighwayEnvRacetrack(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 10,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 80,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.8,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [5, 10],
            "offroad_terminal": False,
            "screen_width": 800,
            "screen_height": 400,
            ##"show_trajectories": True,
        })
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [20,20,20,20,20,20,20,20,20]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[0])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.NONE), width=5, speed_limit=speedlimits[0]))
        net.add_lane("a", "b", StraightLane([42, 10], [100, 10], line_types=(LineType.STRIPED, LineType.NONE), width=5, speed_limit=speedlimits[0]))
        net.add_lane("a", "b", StraightLane([42, 15], [100, 15], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[0]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+10, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1+15, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))
        
        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -18], [120, -32],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -18], [125, -32],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([130, -18], [130, -32],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([135, -18], [135, -32],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))
        
        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+10, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2+15, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3+5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3-5, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3-10, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([63.0868, -22.7], [43.0868, -42.7],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([66.7802, -26.2], [46.7802, -46.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4+5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4+10, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4+15, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4+5, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4+10, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4+15, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5+5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5-5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5-10, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        rng = self.np_random
        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=5,
                lane_from="a",
                lane_to="b",
                lane_id=1,
                spacing=self.config["ego_spacing"],

            )
            MDPvehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            MDPvehicle.MAX_SPEED = 10
            MDPvehicle.MIN_SPEED = 0
            MDPvehicle.trajectory = vehicle.trajectory

            self.controlled_vehicles.append(MDPvehicle)
            self.road.vehicles.append(MDPvehicle)

            from highway_env.vehicle.kinematics  import record_trajectory
            while(len(self.road.vehicles) != others+1):
                random_lane_index = self.road.network.random_lane_index(rng)
                v = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                            longitudinal=rng.uniform(low=0, high=self.road.network.get_lane(random_lane_index).length),
                                            speed=1+rng.uniform(high=3))
                
                v.trajectory.append(record_trajectory(v.road, v.position, v.velocity, v.lane_offset[2], v.heading))
                v.trajectory.append(record_trajectory(v.road, v.position, v.velocity, v.lane_offset[2], v.heading))
                v.trajectory.append(record_trajectory(v.road, v.position, v.velocity, v.lane_offset[2], v.heading))
                dt = 0.25
                #######################
                delta_f = v.action['steering']
                beta = np.arctan(1 / 2 * np.tan(delta_f))

                pred_v1 = v.speed * np.array([np.cos(v.heading + beta), np.sin(v.heading + beta)])
                pred_position1 = v.position + pred_v1*dt
                pred_heading1 = v.heading + v.speed * np.sin(beta) / (v.LENGTH / 2) * dt
                pred_speed1 = v.speed + v.action['acceleration'] * dt
                long, _ = v.lane.local_coordinates(pred_position1)
                pred_angle_offset1 = v.lane.local_angle(pred_heading1, long)
                v.trajectory.append(record_trajectory(v.road, pred_position1, pred_v1, pred_angle_offset1, pred_heading1))

                pred_v2 = pred_speed1 * np.array([np.cos(pred_heading1 + beta), np.sin(pred_heading1 + beta)])
                pred_position2 = pred_position1 + pred_v2*dt
                pred_heading2 = pred_heading1 + pred_speed1 * np.sin(beta) / (v.LENGTH / 2) * dt
                long, _ = v.lane.local_coordinates(pred_position2)
                pred_angle_offset2 = v.lane.local_angle(pred_heading2, long)
                v.trajectory.append(record_trajectory(v.road, pred_position2, pred_v2, pred_angle_offset2, pred_heading2))

                assert len(v.trajectory) == 5
                #################################
                # Prevent early collisions
                for exsiting_v in self.road.vehicles:
                    if np.linalg.norm(v.position - exsiting_v.position) < 10:
                        break
                else:
                    v.MAX_SPEED = 8
                    v.MIN_SPEED = 0
                    v.randomize_behavior()
                    self.road.vehicles.append(v)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        if forward_speed <= 1:
            reward -= 1
        if self.vehicle.crashed or not self.vehicle.on_road:
            reward += self.config["collision_reward"]

        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward
    
    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.time >= self.config["duration"] or not self.vehicle.on_road

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-circle-v0',
    entry_point='highway_env.envs:HighwayEnvCircle',
)

register(
    id='highway-racetrack-v0',
    entry_point='highway_env.envs:HighwayEnvRacetrack',
)