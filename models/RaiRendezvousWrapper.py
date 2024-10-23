import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from deep_rl_for_swarms.ma_envs.commons.utils import EzPickle
from deep_rl_for_swarms.ma_envs import base
# from ma_envs.envs.environment import MultiAgentEnv
from deep_rl_for_swarms.ma_envs.agents.point_agents.rendezvous_agent import PointAgent
from deep_rl_for_swarms.ma_envs.commons import utils as U
import matplotlib.pyplot as plt


class RaiRendezvous(RendezvousEnv):
    def __init__(self, nr_agents=5,
                 obs_mode='sum_obs',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=16,
                 bearing_bins=8,
                 torus=False,
                 dynamics='unicycle',
                 # Here are RAI parameters
                 map_object = None,
                 avoid_direct_collision = True,
                 avoid_buffer_zone = False,
                 avoid_any_damage = False,
                 avoid_max_damage = False):
        super().__init__(nr_agents=5,
                 obs_mode='sum_obs',
                 comm_radius=40,
                 world_size=100,
                 distance_bins=16,
                 bearing_bins=8,
                 torus=False,
                 dynamics='unicycle')
        self.map_object = map_object
        self.avoid_direct_collision = avoid_direct_collision,
        self.avoid_buffer_zone = avoid_buffer_zone,
        self.avoid_any_damage = avoid_any_damage,
        self.avoid_max_damage = avoid_max_damage

    def rai_check_direct_collisions(self, collision_weight=10):
        # simple case: MapObject is a list of point obstacles
        collisions = []
        collision_penalties = []
        for drone in self.world.nodes:
            collisions.append([True if U.get_distances(drone, obstacle) == 0 else False for obstacle in self.map_object])
            collision_penalties.append(sum([collision_weight if collision is True else 0 for collision in collisions]))
        return collisions, collision_penalties


    def get_reward(self, actions):
        total_rai_penalty = []
        if self.avoid_direct_collision:
            colls, coll_penalties = self.rai_check_direct_collisions()
            total_rai_penalty.extend(sum(coll_penalties))

        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)
        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)
        r = - dist_rew - action_pen - sum(total_rai_penalty)
        r = np.ones((self.nr_agents,)) * r
        # print(dist_rew, action_pen)

        return r