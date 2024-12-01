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


def check_status():
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            return f.read().strip() == "pause"
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
                 map_file="map.json"):
        super(OutputWrapper, self).__init__(env)
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
        self.time_limit = 1000,
        self.ob_with_obstacles = []


    def save_model_weights(self):
        weights = self.model.get_weights()  # List of arrays
        with open("model_weights.pkl", "wb") as f:
            pickle.dump(weights, f)
        print("Weights saved for episode:", self.episode)


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

                # Check distance from all obstacles
                # distances_to_obstacles = np.sqrt(
                #     (self.map_data['cflt_x_coord'] - x) ** 2 +
                #     (self.map_data['cflt_y_coord'] - y) ** 2
                # )

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

    def get_collision_data(self):
        """
        Generates data on collisions for use in reward calculations.

        Uses self.map_data, which is the output of MapPackage. For each object in self.map_data, the distance between
        it and each of the drones is calculated using nearest_obstacle_point(), both with and without a buffer.
        The obstacle distance and damage data is then aggregated at a drone-level.

        Returns:
            self.obstacle_df: Dataframe of collision data.
        """
        if len(self.map_data) > 0:

            self.map_data['obstacles'] = [f"{i}-{j}" for i, j in
                                          zip(self.map_data["cstr_obstacle"], self.map_data["cint_obstacle_id"])]
            u_obs = self.map_data['obstacles'].unique()
            obstacle_lookup = {}
            # ob_with_obstacles = []
            # for drone in self.world.nodes:
            #     drone_obst_distance = [nearest_obstacle_point(drone[0], drone[1], self.map_data[self.map_data['obstacles'] == obs]) for obs in u_obs]
            #     ob_with_obstacles.append([i[1] for i in drone_obst_distance])
            # self.ob_with_obstacles = ob_with_obstacles
            for obs in u_obs:
                data = self.map_data[self.map_data['obstacles'] == obs]
                risk = set([row['cflt_obstacle_risk'] for i, row in data.iterrows()])
                if len(risk) == 1:
                    risk = next(iter(risk))  # Get the single value from the set
                else:
                    raise ValueError("Risk set contains multiple values!")
                coords = [(row['cflt_x_coord'], row['cflt_y_coord']) for i, row in data.iterrows()]
                midpoint = [(row['cflt_x_coord'], row['cflt_y_coord']) for i, row in
                            data[data['cstr_point_type'] == 'midpoint'].iterrows()]
                boundaries = [(row['cflt_x_coord'], row['cflt_y_coord']) for i, row in
                              data[data['cstr_point_type'] == 'boundary'].iterrows()]
                obstacle_lookup[obs] = {"risk": risk, "coords": coords, "midpoint": midpoint, "boundaries": boundaries}
            distance_data = []
            for obstacle, data in obstacle_lookup.items():
                # Distance between each drone and obstacle
                obs_distance = [u.nearest_obstacle_point(drone[0], drone[1], data["coords"]) for drone in
                                self.world.nodes]
                obs_distance_norm = u.normalize_distances([i[1] for i in obs_distance])
                # print("obs_distance")
                # print(obs_distance)
                # Collisions: if distance from dists = 0, then collision
                collisions = [1 if dist[1] < 1 else 0 for dist in obs_distance]
                # % of buffer zone entered by each drone
                obs_distance_buffer = [(self.rai_params['buffer_zone_size'] -
                                        u.nearest_obstacle_point(drone[0], drone[1], data["coords"])[1]) /
                                       self.rai_params['buffer_zone_size'] if self.rai_params[
                                                                                  'buffer_zone_size'] > 0 else 0 for
                                       drone in self.world.nodes]
                # print("obs_distance_buffer")
                # print(obs_distance_buffer)
                # Distance between each drone and obstacle boundary
                obs_boundary_distance = [
                    u.nearest_obstacle_point(drone[0], drone[1], data["boundaries"]) if len(data["boundaries"]) > 0 else [
                        data["midpoint"], 1] for drone in self.world.nodes]
                obs_boundary_distance = [ob_bd_dist if collision == 1 else [ob_bd_dist[0], 0] for ob_bd_dist, collision
                                         in zip(obs_boundary_distance, collisions)]
                # print("obs_boundary_distance")
                # print(obs_boundary_distance)
                # Distance between each drone and obstacle midpoint
                obs_midpoint_distance = [u.nearest_obstacle_point(drone[0], drone[1], data["midpoint"]) for drone in
                                         self.world.nodes]
                # print("obs_midpoint_distance")
                # print(obs_midpoint_distance)
                # If collision, distance between midpoint and drone's nearest boundary point
                boundary_midpoint = [u.basic_euclidean(a[0], obstacle_lookup[obstacle]['midpoint']) if c == 1 else 0 for
                                     a, b, c in zip(obs_boundary_distance, obs_distance, collisions)]
                # print("boundary_midpoint")
                # print(boundary_midpoint)

                # >  Calculations
                # # >  Basic Collision Penalties
                collision_pen = [self.rai_params["collision_penalty"] * collision for collision in collisions]
                # # >  Obstacle-Specific Collision Penalties
                risks = [data["risk"] * collision for collision in collisions]
                # # >  Buffer Zone Entry Penalty
                buffer_penalty = [self.rai_params["buffer_entry_penalty"] * dist if dist >= 0 else 0 for dist in
                                  obs_distance_buffer]
                # # > Improvement Multiplier: (distance between drone and boundary)/(distance between midpoint and boundary
                # impr_mult = [round(a / b, 4) for a, b in zip(obs_boundary_distance, boundary_midpoint)]

                df = pd.DataFrame({
                    "cint_drone_id": [i for i, el in enumerate(self.world.nodes)],
                    "cstr_obstacle": [obstacle.split("-")[0] for drone in self.world.nodes],
                    "cstr_obstacle_id": [obstacle.split("-")[1] for drone in self.world.nodes],
                    "cflt_distance_to_obstacle": [dist for dist in obs_distance_norm],
                    "cflt_distance_to_buffer": obs_distance_buffer,
                    "cint_collisions": collisions,
                    "cint_collision_penalties": collision_pen,
                    "cint_risk": risks,
                    "cint_buffer_penalty": buffer_penalty,
                    "cflt_midpoint_dists": [dist[1] for dist in obs_midpoint_distance],
                    # New, not added to db as of 11/15
                    "cflt_nearest_boundary_dist": [dist[1] for dist in obs_boundary_distance],
                    "cflt_midpoint_to_boundary": boundary_midpoint
                })
                distance_data.append(df)
            dfs = pd.concat(distance_data)
            numeric_cols = [col for col in dfs.columns if col.startswith(("cflt", "cint"))]
            dfs[numeric_cols] = dfs[numeric_cols].apply(pd.to_numeric, errors="coerce")
            dfs = dfs.fillna(0)

            self.obstacle_df = dfs.groupby("cint_drone_id", as_index=False).agg(
                {"cflt_distance_to_obstacle": "min",
                 "cint_collisions": "sum",
                 "cint_risk": "sum",
                 "cint_collision_penalties": "sum",
                 "cint_buffer_penalty": "sum",
                 "cflt_midpoint_dists": "sum",
                 "cflt_nearest_boundary_dist": "sum",
                 "cflt_midpoint_to_boundary": "sum"}

            )
            self.obstacle_df["cflt_improvement_multiplier"] = [
                (row["cflt_nearest_boundary_dist"] / row["cflt_midpoint_to_boundary"]) if row[
                                                                                              "cint_collisions"] > 0 else 1
                for i, row in self.obstacle_df.iterrows()]

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
                    "cflt_midpoint_dists": [0 for i, el in enumerate(self.world.nodes)],
                    "cflt_nearest_boundary_dist": [0 for i, el in enumerate(self.world.nodes)],
                    "cflt_midpoint_to_boundary": [0 for i, el in enumerate(self.world.nodes)],
                    "cflt_improvement_multiplier": [0 for i, el in enumerate(self.world.nodes)]
                })

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
                print(f"--- Collisions: {collisions}")
                collision_penalty = [self.rai_params["collision_penalty"] * coll for coll in collisions]
                print(f"--- Penalties: {collision_penalty}")
                distance = [u.check_for_collision(drone, data)[1] for drone in self.world.nodes]
                print(f"--- Distances: {distance}")
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
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)

        distances_to_target = np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1)
        mean_distance_to_target = np.mean(distances_to_target)
        target_distance_reward = -mean_distance_to_target / self.world_size

        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        # RAI penalties
        ## Minimize obstacle proximity
        drone_obstacle_proximity = sum(self.obstacle_df["cflt_distance_to_obstacle"].to_numpy())
        drone_obstacle_proximity_cap = np.where(drone_obstacle_proximity > self.comm_radius, self.comm_radius, drone_obstacle_proximity)
        drone_obstacle_proximity_cap_norm = drone_obstacle_proximity_cap / self.comm_radius
        drone_obstacle_proximity_rew = np.mean(drone_obstacle_proximity_cap_norm)

        ## Avoid Basic Collision
        collision_pen = self.obstacle_df["cint_collision_penalties"].to_numpy()

        ## Avoid Buffer Zone
        all_buffer_penalties = self.obstacle_df["cint_buffer_penalty"].to_numpy()

        # RAI Coefficients
        ## Avoid Collisions?
        avoid_collisions = 1 if self.rai_params["avoid_collisions"]== 1 else 0

        ## Avoid buffer zones?
        avoid_buffer_zones = 1 if self.rai_params["avoid_buffer_zones"] == 1 else 0

        ## Improvement Multiplier
        improvement_multiplier = self.obstacle_df["cflt_improvement_multiplier"].to_numpy()
        collis = self.obstacle_df["cint_collisions"].to_numpy()
        drone_to_bound = self.obstacle_df["cflt_nearest_boundary_dist"].to_numpy()
        mid_to_bound = self.obstacle_df["cflt_midpoint_to_boundary"].to_numpy()

        # Final RAI term
        rai = avoid_collisions*sum((collision_pen)*improvement_multiplier) + avoid_buffer_zones*sum(all_buffer_penalties)

        # r = - dist_rew - action_pen - rai - drone_obstacle_proximity_rew
        r = - target_distance_reward - action_pen - rai - drone_obstacle_proximity_rew
        r = np.ones((self.nr_agents,)) * r

        return r

    def rai_reward_data(self, actions):
        self.get_collision_data2()
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius

        distances_to_target = np.linalg.norm(self.world.agent_states[:, :2] - self.target_point, axis=1)
        mean_distance_to_target = np.mean(distances_to_target)
        target_distance_reward = -mean_distance_to_target / self.world_size

        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        # RAI penalties
        ## Minimize obstacle proximity
        drone_obstacle_proximity = sum(self.obstacle_df["cflt_distance_to_obstacle"].to_numpy())
        drone_obstacle_proximity_cap = np.where(drone_obstacle_proximity > self.comm_radius, self.comm_radius,
                                                drone_obstacle_proximity)
        drone_obstacle_proximity_cap_norm = drone_obstacle_proximity_cap / self.comm_radius
        drone_obstacle_proximity_rew = np.mean(drone_obstacle_proximity_cap_norm)

        ## Avoid Basic Collision
        collisions = self.obstacle_df["cint_collisions"].to_numpy()
        obs_distance = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
        collision_pen = self.obstacle_df["cint_collision_penalties"].to_numpy()

        ## Avoid Buffer Zone
        all_buffer_penalties = self.obstacle_df["cint_buffer_penalty"].to_numpy()

        # RAI Coefficients
        ## Avoid Collisions?
        avoid_collisions = 1 if self.rai_params["avoid_collisions"] == True else 0

        ## Avoid buffer zones?
        avoid_buffer_zones = 1 if self.rai_params["avoid_buffer_zones"] == True else 0

        ## Improvement Multiplier
        improvement_multiplier = self.obstacle_df["cflt_improvement_multiplier"].to_numpy()

        ## Drone Damage
        drone_damage = self.obstacle_df["cint_collision_penalties"].to_numpy()
        # Final RAI term
        rai =  avoid_collisions * sum((collision_pen) * improvement_multiplier) + avoid_buffer_zones * sum(all_buffer_penalties)

        rai_data = {
            "cflt_distance_reward": dist_rew,
            "cflt_target_distance_reward": target_distance_reward,
            "cflt_action_penalty": action_pen,
            "cflt_rai_reward": rai,
            "cflt_basic_collision_penalty": collision_pen,
            "cint_buffer_zone_entry_penalty": all_buffer_penalties,
            "cint_all_collisions": collisions,
            "cflt_obstacle_distance": obs_distance,
            "cflt_improvement_multiplier": improvement_multiplier,
            "cflt_drone_damage": drone_damage
        }
        return rai_data


    def step(self, action):
        state = self.env.state if hasattr(self.env, "state") else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        reward = self.get_rai_reward(action)
        rai_reward = self.rai_reward_data(action)

        check_status()
        print("----------------------------------------------")
        print(f"Reward: {reward}")
        print(f"RAI: {rai_reward['cflt_rai_reward']}")
        print(f"Collision Penalty: {rai_reward['cflt_basic_collision_penalty']}")
        print(f"Buffer Penalty: {rai_reward['cint_buffer_zone_entry_penalty']}")
        print(f"All Collisions: {rai_reward['cint_all_collisions']}")
        print(f"Obstacle Distance: {rai_reward['cflt_obstacle_distance']}")
        print(f"Improvement Multiplier: {rai_reward['cflt_improvement_multiplier']}")
        print(f"Drone Damage: {rai_reward['cflt_drone_damage']}")



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
            "cflt_target_distance_reward": rai_reward["cflt_target_distance_reward"],
            "cflt_action_penalty": rai_reward["cflt_action_penalty"],
            "cint_all_collisions": rai_reward["cint_all_collisions"],
            "cflt_drone_obstacle_distance": rai_reward["cflt_obstacle_distance"],
            "cflt_improvement_multiplier": rai_reward["cflt_improvement_multiplier"],
            "cflt_drone_damage": rai_reward["cflt_drone_damage"]
        }

        self.log_data.append(log_entry)
        self.episode_rewards.append(reward)
        #self.run_data.append(run_entry)
        self.episode += 1
        update_episode(self.episode)
        print(f'On episode: {self.episode}', flush=True)
        if check_status():
            print(f"Model paused at episode {self.episode}.", flush=True)
            self.batched_commits()
            update_db("commit")
            while check_status():
                time.sleep(1)  # Wait until pause flag is cleared
            print("Model resumed.", flush=True)

        # Batch output for mid-run database commits
        if self.episode == 1:
            self.run_parameters()

        if self.episode % 10 == 0:
            if self.episode == 10:
                self.batched_commits()
            else:
                self.batched_commits(first=False)
            update_db("commit")

        # time.sleep(0.25)
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
        update_episode(0)
        update_db("commit")
        super(OutputWrapper, self).close()

    def batched_commits(self, first=True):
        print("Batching output....")
        if first:
            print(self.map_data)
            out = xo.OutputObject(self.log_data, self.param_data, self.all_map_data, self.rai_data)
            out.generate_tables()
            out.pickle_tables()
            self.log_data = []
        else:
            out = xo.OutputObject(self.log_data, [], [], [])
            out.generate_tables()
            out.pickle_tables()
            self.log_data = []
        print("Batch is pickled")



