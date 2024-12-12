import datetime
import gym
import numpy as np
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy
from models.drlss.deep_rl_for_swarms.ma_envs.commons import utils as U
import time
import os
import utils.app_utils as u
import XRAI_Output as xo

# -------------------------------------------------------------------------- #
# ---------------------------- Description --------------------------------- #
# -------------------------------------------------------------------------- #
# This script contains two classes and a set of helper functions
# > Helper functions
# > OutputWrapper:
#   > Input: DRL models
#   > Output: JSON/dictionaries of DRL output.
#   > Description: This is an edited version of the output wrapper we"ve been using.
#       All extraneous or redundant information has been removed.
# > OutputObject:
#   > Input: JSON/dictionaries created by DRLOutput
#   > Output: Dictionary of tables corresponding to the tables defined in
#       "DRL Data Dictionary.xlsx"; can also bulk append output to database;
#       pickled file of dictionary of tables
#   > Description: Flattens the JSON from DRLOutput into DataFrames that will be
#       added to the API database (\app\data_models.py)

# -------------------------------------------------------------------------- #
# ------------------------ Helper Functions -------------------------------- #
# -------------------------------------------------------------------------- #

api_base_url = os.getenv('API_BASE_URL')
db_status_path = "/CapstoneTeamAPI/utils/status/data_commit.txt"
episode_path = "/CapstoneTeamAPI/utils/status/model_episode.txt"
status_path = "/CapstoneTeamAPI/utils/status/model_status.txt"


# -------------------------------------------------------------------------- #
# ------------------------ Helper Functions -------------------------------- #
# -------------------------------------------------------------------------- #


def check_pause():
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            return f.read().strip() == "pause"
    else:
        return False

def check_stop():
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            return f.read().strip() == "stop"
    else:
        return False

def reset_status():
    if os.path.exists(status_path):
        with open(status_path, "w") as f:
            return f.write("complete")
    return False

def update_db(status):
    if os.path.exists(db_status_path):
        with open(db_status_path, "w") as f:
            return f.write(status)
    return False

def update_episode(episode):
    if os.path.exists(episode_path):
        with open(episode_path, "w") as f:
            return f.write(str(episode))
    return False


db = SQLAlchemy()

# ------------------------------------------------------------------------------- #
# ------------- OutputWrapper(ModelObject): models output wrapper --------------- #
# ------------------------------------------------------------------------------- #


class OutputWrapper(gym.Wrapper):
    def __init__(self, env, env_type, environment_params, model_params, map_object,
                 rai_params=None,
                 log_file="output.json",
                 param_file="param.json",
                 rai_file="rai.json",
                 map_file="map.json",
                 run_type = "app"):
        super(OutputWrapper, self).__init__(env)
        self.run_type = run_type
        self.run_date = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S")
        self.env_type = env_type
        self.environment_params = environment_params
        self.rai_params = rai_params
        self.model_params = model_params
        self.log_file = log_file
        self.param_file = param_file
        self.rai_file = rai_file
        self.map_file = map_object
        self.episode_rewards = []
        self.episode = 0
        self.log_data = []
        self.param_data = []
        self.rai_data = pd.DataFrame([self.rai_params])
        self.run_data = []
        self.all_map_data = map_object
        self.map_data = map_object[map_object["cstr_obstacle"] != 'target']
        self.obstacle_df = None
        self.weights_data = []
        self.run_status = "running"
        target_x = map_object[map_object["cstr_obstacle"] == 'target']['cflt_midpoint_x_coord'].iloc[0]
        target_y = map_object[map_object["cstr_obstacle"] == 'target']['cflt_midpoint_x_coord'].iloc[0]
        self.target_point = np.array([target_x, target_y])
        self.time_limit = 30,
        self.ob_with_obstacles = []

    def status_for_db(self):
        if self.episode == 0:
            return "model run initiated"
        elif check_pause():
            return "paused"
        elif check_stop():
            return "stopped"
        else:
            return "active"

    @property
    def timestep_limit(self):
        return self.time_limit

    def run_parameters(self, **kwargs):
        env_entry = {de: val for de, val in self.environment_params.items()}
        model_entry = {de: val for de, val in self.model_params.items()}
        param_entry = {"environment_id": self.env_type, **env_entry, **model_entry}
        self.param_data.append(param_entry)

    def reset(self, **kwargs):
        self.episode_rewards = []
        self.episode = 0
        agent_states = []
        for _ in range(self.env.nr_agents):
            while True:
                # Generate random (x, y) position for the drone
                x = self.env.world_size * ((0.95 - 0.05) * np.random.rand() + 0.05)
                y = self.env.world_size * ((0.95 - 0.05) * np.random.rand() + 0.05)

                distances_to_obstacles = [u.check_for_collision((x,y), row)[1] for i, row in self.map_data.iterrows()]

                if np.all(np.array(distances_to_obstacles) >= 1):
                    orientation = 2 * np.pi * np.random.rand()  # Random orientation
                    agent_states.append([x, y, orientation])
                    break  # Exit loop for this drone

        # Assign valid positions to the environment's agent states
        self.env.world.agent_states = np.array(agent_states)

        # Call the original reset() to initialize other components
        obs = self.env.reset(**kwargs)

        # Ensure the manually assigned positions remain intact
        self.env.world.agent_states = np.array(agent_states)

        return obs

    def get_collision_data2(self):
        if len(self.map_data) > 0:
            self.map_data['obstacles'] = [f"{i}-{j}" for i, j in
                                          zip(self.map_data["cstr_obstacle"], self.map_data["cint_obstacle_id"])]
            u_obs = self.map_data['obstacles'].unique()
            all_obstacle_data = []
            for obs in u_obs:
                #print(f"Obstacle: {obs}")
                data = self.map_data[self.map_data['obstacles'] == obs]
                #print(f"--- Obstacle Data: {data}")
                drones = [i for i, el in enumerate(self.world.nodes)]
                obstacles = [obs for drone in self.world.nodes]
                risk = [data['cflt_obstacle_risk'].iloc[0] for drone in self.world.nodes]
                collisions = [1 if u.check_for_collision(drone, data)[0] else 0 for drone in self.world.nodes]
                # print(f"--- Collisions: {collisions}")
                collision_penalty = [self.rai_params["collision_penalty"] * coll for coll in collisions]
                # print(f"--- Penalties: {collision_penalty}")
                distance = [u.check_for_collision(drone, data)[1] for drone in self.world.nodes]
                # print(f"--- Distances: {distance}")
                buffer_distance = [0 if u.check_for_collision(drone, data)[1] > self.rai_params['buffer_zone_size'] else u.check_for_collision(drone, data)[1] for drone in self.world.nodes]
                buffer_penalty = [(self.rai_params['buffer_zone_size'] - buff_dist)/self.rai_params['buffer_zone_size'] if buff_dist > 0 else 0 for buff_dist in buffer_distance]
                dist_to_boundary = [u.check_for_collision(drone, data)[2] if u.check_for_collision(drone, data)[0] else 0 for drone in self.world.nodes]
                dist_to_midpoint = [u.check_for_collision(drone, data)[3] if u.check_for_collision(drone, data)[0] else 0 for drone in self.world.nodes]
                midpoint_to_boundary = [b + m if b > 1 and m > 1 else 1 for b,m in zip(dist_to_boundary, dist_to_midpoint)]
                obs_df = pd.DataFrame({
                    "cint_drone_id": drones,
                    "cstr_obstacle": obstacles,
                    "cflt_distance_to_obstacle": distance,
                    "cflt_distance_to_buffer": buffer_distance,
                    "cint_collisions": collisions,
                    "cint_collision_penalties": collision_penalty,
                    "cint_buffer_penalty": buffer_penalty,
                    "cint_risk": risk,
                    "cflt_nearest_boundary_dist": dist_to_boundary,
                    "cflt_midpoint_to_boundary": midpoint_to_boundary
                })
                all_obstacle_data.append(obs_df)
            dfs = pd.concat(all_obstacle_data)
            numeric_cols = [col for col in dfs.columns if col.startswith(("cflt", "cint"))]
            dfs[numeric_cols] = dfs[numeric_cols].apply(pd.to_numeric, errors="coerce")
            dfs = dfs.fillna(0)
            #print("Printing dfs....")
            #print(dfs["cint_risk"])
            self.obstacle_df = dfs.groupby("cint_drone_id", as_index=False).agg(
                {"cflt_distance_to_obstacle": "min",
                 "cint_collisions": "sum",
                 "cint_risk": "mean",
                 "cint_collision_penalties": "sum",
                 "cint_buffer_penalty": "sum",
                 #"cflt_midpoint_dists": "sum",
                 "cflt_nearest_boundary_dist": "sum",
                 "cflt_midpoint_to_boundary": "sum"
                 }

            )

            self.obstacle_df["cflt_improvement_multiplier"] = [
                (row["cflt_nearest_boundary_dist"] / row["cflt_midpoint_to_boundary"])  if row[
                                                                                              "cint_collisions"] > 0 else 1 for i, row in self.obstacle_df.iterrows()]

            self.obstacle_df["cflt_improvement_multiplier"] = self.obstacle_df["cflt_improvement_multiplier"].fillna(0)
            self.obstacle_df.replace([np.inf, -np.inf], 1, inplace=True)
            self.map_data = self.map_data.drop('obstacles', axis=1)
        else:
            self.obstacle_df = pd.DataFrame({
                "cint_drone_id": [i for i, el in enumerate(self.world.nodes)],
                "cflt_distance_to_obstacle": [0 for i, el in enumerate(self.world.nodes)],
                "cflt_distance_to_buffer": [0 for i, el in enumerate(self.world.nodes)],
                "cint_collisions": [0 for i, el in enumerate(self.world.nodes)],
                "cint_collision_penalties": [0 for i, el in enumerate(self.world.nodes)],
                "cint_risk": [0 for i, el in enumerate(self.world.nodes)],
                "cint_buffer_penalty": [0 for i, el in enumerate(self.world.nodes)],
                #"cflt_midpoint_dists": [0 for i, el in enumerate(self.world.nodes)],
                "cflt_nearest_boundary_dist": [0 for i, el in enumerate(self.world.nodes)],
                "cflt_midpoint_to_boundary": [0 for i, el in enumerate(self.world.nodes)],
                "cflt_improvement_multiplier": [0 for i, el in enumerate(self.world.nodes)]
            })

    def get_rai_reward(self, actions):
        self.get_collision_data2()
        # Interdrone Distance
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius
        dist_rew = np.mean(all_distances_cap_norm)
        var_dist_rew = np.var(all_distances)
        action_pen = 0.001 * np.mean(actions ** 2)


        # Distance to Target
        distances_to_target = np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1)
        distances_to_target_cap = np.where(distances_to_target > self.comm_radius, self.comm_radius,
                                           distances_to_target)
        distances_to_target_cap_norm = distances_to_target_cap / self.comm_radius
        mean_distance_to_target = np.mean(distances_to_target_cap_norm)
        target_distance_reward = mean_distance_to_target / self.world_size

        # Directional Reward -- how well are the drones' trajectories aligned to the target?
        # Directional Reward with Smoothing
        direction_to_target = self.target_point - self.world.agent_states[:, :2]
        direction_norm = np.linalg.norm(direction_to_target, axis=1) + 1e-6
        direction_unit_vector = direction_to_target / direction_norm[:, None]

        # Smooth actions by normalizing
        actions_norm = np.linalg.norm(actions[:, :2], axis=1) + 1e-6
        actions_unit_vector = actions[:, :2] / actions_norm[:, None]

        # Compute alignment and smooth
        alignment = np.sum(actions_unit_vector * direction_unit_vector, axis=1)
        #directional_reward = np.mean(alignment)

        # Angular Deviation Penalty
        angles = np.arccos(np.clip(np.sum(actions_unit_vector * direction_unit_vector, axis=1), -1.0, 1.0))
        angular_penalty = np.mean(angles)  # Smaller angles mean better alignment

        # Reward term (negative penalty)
        directional_reward = -angular_penalty

        # Target Reached Bonus
        threshold = 1.0  # Define a threshold for "reaching" the target
        target_reached = np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1) < threshold
        target_bonus = np.sum(target_reached) * 10

        proximity_bonus = np.mean(np.exp(-0.5 * np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1)))

        # Obstacle Proximity Penalty
        obstacle_penalty = np.mean(
        np.maximum(0, self.rai_params['buffer_zone_size'] - self.obstacle_df["cflt_distance_to_obstacle"]))

        # RAI
        ## Minimize obstacle proximity
        drone_obstacle_proximity = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
        drone_obstacle_proximity_cap = np.where(drone_obstacle_proximity > self.comm_radius, self.comm_radius,
                                                drone_obstacle_proximity)
        drone_obstacle_proximity_cap_norm = drone_obstacle_proximity_cap / self.comm_radius
        drone_obstacle_proximity_rew = np.mean(drone_obstacle_proximity_cap_norm)

        ## Avoid Collisions?
        avoid_collisions = 1 if self.rai_params["avoid_collisions"] == True else 0
        ## Collision Penalty
        collisions = self.obstacle_df["cint_collisions"].to_numpy()
        obs_distance = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
        collision_pen = self.obstacle_df["cint_collision_penalties"].to_numpy()
        collision_penalty = np.sum(self.obstacle_df["cint_collision_penalties"])
        ## Avoid buffer zones?
        avoid_buffer_zones = 1 if self.rai_params["avoid_buffer_zones"] == True else 0
        ## Buffer Zone Penalty
        buffer_penalty = np.sum(self.obstacle_df["cint_buffer_penalty"])


        target_r = - (0.5 * target_distance_reward) - (0.5 * var_dist_rew)
        rai_r = (collision_penalty * avoid_collisions) + (buffer_penalty * avoid_buffer_zones)
        # Combine Rewards
        r = target_r - rai_r
        # Per-Agent Reward
        r = np.ones((self.nr_agents,)) * r
        return r

    def rai_reward_data(self, actions):
        self.get_collision_data2()
        # Interdrone Distance
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius
        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        # Distance to Target
        distances_to_target = np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1)
        distances_to_target_cap = np.where(distances_to_target > self.comm_radius, self.comm_radius,
                                           distances_to_target)
        distances_to_target_cap_norm = distances_to_target_cap / self.comm_radius
        mean_distance_to_target = np.mean(distances_to_target_cap_norm)
        target_distance_reward = mean_distance_to_target / self.world_size

        # Directional Reward -- how well are the drones' trajectories aligned to the target?
        direction_to_target = self.target_point - self.world.agent_states[:, :2]
        direction_norm = np.linalg.norm(direction_to_target, axis=1) + 1e-6
        direction_norm = direction_norm[:, None]
        directional_reward = np.mean(np.sum((actions[:, :2] * direction_to_target) / direction_norm, axis=1))

        # Target Reached Bonus
        threshold = 1.0  # Define a threshold for "reaching" the target
        target_reached = np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1) < threshold
        target_bonus = np.sum(target_reached) * 10

        # Obstacle Proximity Penalty
        obstacle_penalty = np.mean(
            np.maximum(0, self.rai_params['buffer_zone_size'] - self.obstacle_df["cflt_distance_to_obstacle"]))

        # RAI
        ## Minimize obstacle proximity
        drone_obstacle_proximity = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
        drone_obstacle_proximity_cap = np.where(drone_obstacle_proximity > self.comm_radius, self.comm_radius,
                                                drone_obstacle_proximity)
        drone_obstacle_proximity_cap_norm = drone_obstacle_proximity_cap / self.comm_radius
        drone_obstacle_proximity_rew = np.mean(drone_obstacle_proximity_cap_norm)
        ## Avoid Collisions?
        avoid_collisions = 1 if self.rai_params["avoid_collisions"] == True else 0
        ## Collision Penalty
        collisions = self.obstacle_df["cint_collisions"].to_numpy()
        obs_distance = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
        collision_pen = self.obstacle_df["cint_collision_penalties"].to_numpy()
        collision_penalty = np.sum(self.obstacle_df["cint_collision_penalties"])
        ## Avoid buffer zones?
        avoid_buffer_zones = 1 if self.rai_params["avoid_buffer_zones"] == True else 0
        ## Buffer Zone Penalty
        buffer_penalty = np.sum(self.obstacle_df["cint_buffer_penalty"])
        ## Improvement Multiplier
        improvement_multiplier = self.obstacle_df["cflt_improvement_multiplier"].to_numpy()

        ## Drone Damage
        drone_damage = self.obstacle_df["cint_collision_penalties"].to_numpy()
        # Final RAI term
        rai =  (collision_penalty * avoid_collisions) - (buffer_penalty * avoid_buffer_zones)

        rai_data = {
            "cflt_distance_reward": dist_rew,
            "cflt_direction_reward": directional_reward,
            "cflt_target_distance_reward": target_distance_reward,
            "cflt_action_penalty": action_pen,
            "cflt_rai_reward": rai,
            "cflt_basic_collision_penalty": (collision_penalty * avoid_collisions),
            "cint_buffer_zone_entry_penalty": (buffer_penalty * avoid_buffer_zones),
            "cint_all_collisions": collisions,
            "cflt_obstacle_distance": drone_obstacle_proximity_cap_norm,
            "cflt_improvement_multiplier": improvement_multiplier,
            "cflt_drone_damage": drone_damage
        }
        return rai_data


    def step(self, action):
        state = self.env.state if hasattr(self.env, "state") else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        reward = self.get_rai_reward(action)
        rai_reward = self.rai_reward_data(action)


        log_entry = {
            "cint_n_drones": self.env.nr_agents,
            "cint_episode": self.episode,
            "cdtm_run_date": self.run_date,
            "cbln_terminal": done,
            "cstr_global_state": state,
            "local_state": info["state"],
            "actions": info["actions"],
            "cflt_reward": reward,
            "cflt_distance_reward": rai_reward["cflt_distance_reward"],
            "cflt_direction_reward": rai_reward["cflt_direction_reward"],
            "cflt_target_distance_reward": rai_reward["cflt_target_distance_reward"],
            "cflt_action_penalty": rai_reward["cflt_action_penalty"],
            "cflt_collision_penalty": rai_reward["cflt_basic_collision_penalty"],
            "cflt_buffer_penalty": rai_reward["cint_buffer_zone_entry_penalty"],
            "cint_all_collisions": rai_reward["cint_all_collisions"],
            "cflt_drone_obstacle_distance": rai_reward["cflt_obstacle_distance"],
            "cflt_improvement_multiplier": rai_reward["cflt_improvement_multiplier"],
            "cflt_drone_damage": rai_reward["cflt_drone_damage"],
            "cstr_model_status": self.status_for_db()
        }

        self.log_data.append(log_entry)
        self.episode_rewards.append(reward)

        self.episode += 1
        update_episode(self.episode)
        print(f'On episode: {self.episode}', flush=True)
        if check_pause():
            print(f"Model paused at episode {self.episode - 1}.", flush=True)
            self.batched_commits()
            update_db("commit")
            while check_pause():
                time.sleep(1)  # Wait until pause flag is cleared
            print("Model resumed.", flush=True)
        if check_stop():
            print(f"Model stopped at episode {self.episode - 1}.", flush=True)
            self.batched_commits()
            update_db("commit")

        # Batch output for mid-run database commits
        if self.run_type == "app":
            if self.episode == 1:
                self.run_parameters()

            if self.episode % 10 == 0:
                if self.episode == 10:
                    self.batched_commits()
                else:
                    self.batched_commits(first=False)
                update_db("commit")
        elif self.run_type == "api":
            if self.episode == self.time_limit:
                self.batched_commits(first=False)
                update_db("commit")


        #time.sleep(0.25)
        return next_state, reward, done, info

    def close(self):
        # Commit any remaining data
        output = xo.OutputObject(self.log_data,
                              [],
                              [],
                              [])
        output.generate_tables()
        output.pickle_tables()
        reset_status()

        try:
            self.batched_commits(first=False)
            print("Final data committed to the database.")
        except Exception as e:
            print(f"Error committing final data: {str(e)}")

        update_episode(0)
        update_db("commit")
        super(OutputWrapper, self).close()


    def batched_commits(self, first=True):
        print("Batching output....")
        if first:
            out = xo.OutputObject(self.log_data, self.param_data, self.all_map_data, self.rai_data)
            out.generate_tables()
            out.pickle_tables()

        else:
            out = xo.OutputObject(self.log_data, [], [], [])
            out.generate_tables()
            out.pickle_tables()

        self.log_data = []
        print("Batch is pickled")



