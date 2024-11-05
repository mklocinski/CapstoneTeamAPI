import datetime
import gym
import numpy as np
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy
from models.drlss.deep_rl_for_swarms.ma_envs.commons import utils as U
import requests
import time

api_base_url = "https://xraiapi-ba66c372be3f.herokuapp.com/api"

def get_run_id():
    response = requests.get("http://app:8000/api/get_run_id")
    if response.status_code == 200:
        return response.json().get("id")
    else:
        print("Error checking model status:", response.json().get("error"))
        return None

def check_model_status():
    response = requests.get(f"{api_base_url}/check_model_status")
    if response.status_code == 200:
        return response.json().get("status")
    else:
        print("Error checking model status:", response.json().get("error"))
        return None


def record_model_episode(current_episode):
    url = f"{api_base_url}/record_model_episode"
    response = requests.post(url, json={"current_episode": current_episode})

    if response.status_code == 200:
        print("Episode recorded successfully.")
    else:
        print("Error recording episode:", response.json().get("error"))

db = SQLAlchemy()
# -------------------------------------------------------------------------- #
# ---------------------------- Description --------------------------------- #
# -------------------------------------------------------------------------- #
# This script contains two classes and a set of helper functions
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
# > Helper functions
#   > scalarize_[insert table](): scalarizes pickled tables for loading into
#     database

# ------------------------------------------------------------------------------- #
# ------------- OutputWrapper(ModelObject): models output wrapper --------------- #
# ------------------------------------------------------------------------------- #

def nearest_obstacle_point(x,y, xy_array):
    all_distances = np.sqrt((xy_array[:, 0] - x) ** 2 + (xy_array[:, 1] - y) ** 2)
    min_distance = min(all_distances)
    nearest = xy_array[np.argmin(all_distances)]
    return min_distance


class OutputWrapper(gym.Wrapper):
    def __init__(self, env, env_type, environment_params, model_params, map_object,
                 rai_params=None,
                 log_file="output.json",
                 param_file="param.json",
                 rai_file="rai.json",
                 map_file="map.json",
                 run_type="standard"):
        super(OutputWrapper, self).__init__(env)
        self.run_date = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S")
        self.run_type = run_type
        self.env_type = env_type
        self.environment_params = environment_params
        self.rai_params = rai_params
        self.model_params = model_params
        self.log_file = log_file
        self.param_file = param_file
        self.rai_file = rai_file
        self.map_file = map_file
        self.episode_rewards = []
        self.episode = 0
        self.log_data = []
        self.param_data = []
        self.rai_data = []
        self.run_data = []
        self.map_data = map_object
        self.obstacle_df = None
        self.run_type = run_type

    def run_parameters(self, **kwargs):
        env_entry = {de: val for de, val in self.environment_params.items()}
        model_entry = {de: val for de, val in self.model_params.items()}
        param_entry = {"environment_id": self.env_type, **env_entry, **model_entry}
        self.param_data.append(param_entry)

    def reset(self, **kwargs):
        """Resets the environment and initializes logging for a new episode."""
        self.episode_rewards = []
        self.episode = 0
        return self.env.reset(**kwargs)

    def get_collision_data(self):
        obstacles = self.map_data["cstr_obstacle"].unique()

        obstacle_lookup = {obs: self.map_data[self.map_data["cstr_obstacle"]==obs]["cint_obstacle_id"].unique() for obs in obstacles}
        distance_data = []
        for obstacle in obstacles:
            obstacle_risk = self.map_data[self.map_data["cstr_obstacle"]==obstacle]["cint_obstacle_risk"].unique()
            for id in obstacle_lookup[obstacle]:
                df = self.map_data[(self.map_data["cstr_obstacle"] == obstacle) &
                                   (self.map_data["cint_obstacle_id"] == id) &
                                   (self.map_data["cstr_point_type"] == "boundary")]
                boundary = np.array([(x, y) for x, y in zip(df["cflt_x_coord"], df["cflt_y_coord"])])
                dists = [nearest_obstacle_point(drone[0], drone[1], boundary) for drone in self.world.nodes]
                collisions = [1 if dist == 0 else 0 for dist in dists]
                risks = [obstacle_risk if dist == 0 else 0 for dist in dists]
                df = pd.DataFrame({
                    "cint_drone_id":[i for i, el in enumerate(self.world.nodes)],
                    "cstr_obstacle":[obstacle for drone in self.world.nodes],
                    "cstr_obstacle_id": [id for drone in self.world.nodes],
                    "cflt_distance_to_obstacle":dists,
                    "cint_collisions":collisions,
                    "cint_risk": risks
                })
                distance_data.append(df)
        dfs = pd.concat(distance_data)
        self.obstacle_df = dfs.groupby("cint_drone_id", as_index=False).agg(
            {"cflt_distance_to_obstacle":"sum",
             "cint_collisions":"sum",
             "cint_risk":"sum"}
        )
    def get_rai_reward(self, actions):
        self.get_collision_data()
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)

        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        if self.rai_params["basic_collision_avoidance"]==True:
            all_collisions = self.obstacle_df["cint_collisions"].to_numpy()
            collision_pen = sum([100 * collision for collision in all_collisions])
            all_obstacle_distances = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
            all_obstacles_cap = np.where(all_obstacle_distances > self.comm_radius, self.comm_radius,
                                         all_obstacle_distances)
            all_obstacles_cap_norm = all_obstacles_cap / self.comm_radius
            r = - dist_rew - action_pen - collision_pen
            r = np.ones((self.nr_agents,)) * r
        else:
            r = - dist_rew - action_pen
            r = np.ones((self.nr_agents,)) * r

        return r

    def rai_reward_data(self, actions):
        self.get_collision_data()
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)

        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        r = - dist_rew - action_pen

        if self.rai_params["basic_collision_avoidance"]==True:
            all_collisions = self.obstacle_df["cint_collisions"].to_numpy()
            collision_pen = sum([100 * collision for collision in all_collisions])
            all_obstacle_distances = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
            all_obstacles_cap = np.where(all_obstacle_distances > self.comm_radius, self.comm_radius,
                                         all_obstacle_distances)
            all_obstacles_cap_norm = all_obstacles_cap / self.comm_radius
            r_with_collisions = - dist_rew - action_pen - collision_pen
        else:
            r_with_collisions = None
            all_collisions = None
            all_obstacles_cap = None

        rai_data = {
            "cflt_distance_reward": dist_rew,
            "cflt_action_penalty": action_pen,
            "cflt_reward_with_collisions": r_with_collisions,
            "cint_all_collisions": all_collisions,
            "cflt_capped_obstacle_distance": all_obstacles_cap
        }
        return rai_data

    def step(self, action):
        state = self.env.state if hasattr(self.env, "state") else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        reward = self.get_rai_reward(action)
        rai_reward = self.rai_reward_data(action)

        if self.run_type == "live":
            run_entry = {
                    "cint_episode": self.episode,
                    "cstr_run_status": check_model_status(),
                    "cdtm_status_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        else:
            run_entry = {
                "cint_episode": self.episode,
                "cstr_run_status": "running",
                "cdtm_status_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

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
            "cflt_action_penalty": rai_reward["cflt_action_penalty"],
            "cint_drone_collisions": rai_reward["cint_all_collisions"],
            "cflt_drone_obstacle_distance": rai_reward["cflt_capped_obstacle_distance"]
        }

        self.log_data.append(log_entry)
        self.episode_rewards.append(reward)
        self.run_data.append(run_entry)
        self.episode += 1
        print(f'On episode: {self.episode}', flush=True)
        if self.run_type == "live":
            record_model_episode(self.episode)

        return next_state, reward, done, info

    def convert_ndarray(self, item):
        """Recursively convert numpy arrays to lists."""
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: self.convert_ndarray(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self.convert_ndarray(i) for i in item]
        else:
            return item

    def close(self):
        self.run_parameters()
        serializable_param_data = self.convert_ndarray(self.param_data)
        serializable_log_data = self.convert_ndarray(self.log_data)


        # # Writing to file is commented out
        # with open(self.param_file, "w") as f:
        #     json.dump(serializable_param_data, f, indent=4)
        # with open(self.log_file, "w") as f:
        #     json.dump(serializable_log_data, f, indent=4)
        # with open(self.map_file, "w") as f:
        #     json.dump(self.map_data)

        super(OutputWrapper, self).close()

    def batched_commits(self, n=100):
        if len(self.param_data) == n:
            print("Batching output....")
            out = OutputObject(self.log_data, self.param_data, self.map_data)
            out.generate_tables()
            out.pickle_tables()
            self.log_data, self.param_data, self.map_data = [], [], []
            print("Batch is pickled")
        return


# -------------------------------------------------------------------------- #
# -------------- OutputObject: converts output to DataFrames --------------- #
# -------------------------------------------------------------------------- #
class OutputObject:
    def __init__(self, output, params, map, run_data = None, output_type="json"):
        self.output = output
        self.params = params[0]
        self.run_data = run_data
        self.map = map
        self.output_type = output_type
        self.tables = {}


    def make_tbl_model_runs(self):
        start_time = time.time()

        if len(self.output) > 0:
            run_date = self.output[0]["cdtm_run_date"]
            terminal_episode = [self.output[i]["cbln_terminal"] for i in range(len(self.output))].index(True)
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_model_runs duration: {query_duration:.2f} seconds")
            return pd.DataFrame({"cdtm_run_date": [run_date], "cbln_terminal_episode": [terminal_episode]})

    def make_tbl_model_run_params(self):
        start_time = time.time()

        if len(self.params) > 0:
            ps = {'cstr_environment_id': self.params['environment_id'],
                  'cint_nr_agents': self.params['nr_agents'],
                  'cstr_obs_mode': self.params['obs_mode'],
                  'cint_comm_radius': self.params['comm_radius'],
                  'cint_world_size': self.params['world_size'],
                  'cint_distance_bins': self.params['distance_bins'],
                  'cint_bearing_bins': self.params['bearing_bins'],
                  'cbln_torus': self.params['torus'],
                  'cstr_dynamics': self.params['dynamics'],
                  'cint_timesteps_per_batch': self.params['timesteps_per_batch'],
                  'cflt_max_kl': self.params['max_kl'],
                  'cint_cg_iters': self.params['cg_iters'],
                  'cflt_cg_damping': self.params['cg_damping'],
                  'cflt_gamma': self.params['gamma'],
                  'cflt_lam': self.params['lam'],
                  'cint_vf_iters': self.params['vf_iters'],
                  'cint_vf_stepsize': self.params['vf_stepsize']}
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_model_run_params duration: {query_duration:.2f} seconds")
            return pd.DataFrame(ps, index=[0])

    def make_tbl_local_state(self):
        start_time = time.time()

        if len(self.output) > 0:
            episodes = []
            drones = []
            x_coords = []
            y_coords = []
            orientation = []
            linear_velocity = []
            angular_velocity = []
            collisions = []
            obstacle_distances = []
            for episode in range(len(self.output)):
                for drone in range(self.output[0]["cint_n_drones"]):
                    episodes.append(episode)
                    drones.append(drone)
                    x_coords.append(self.output[episode]["local_state"][drone][0])
                    y_coords.append(self.output[episode]["local_state"][drone][1])
                    orientation.append(self.output[episode]["local_state"][drone][2])
                    linear_velocity.append(self.output[episode]["local_state"][drone][3])
                    angular_velocity.append(self.output[episode]["local_state"][drone][4])
                    collisions.append(self.output[episode]["cint_drone_collisions"][drone])
                    obstacle_distances.append(self.output[episode]["cflt_drone_obstacle_distance"][drone])
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_local_state duration: {query_duration:.2f} seconds")
            return pd.DataFrame({
                "cint_episode_id": episodes,
                "cint_drone_id": drones,
                "cflt_x_coord": x_coords,
                "cflt_y_coord": y_coords,
                "cflt_orientation": orientation,
                "cflt_linear_velocity": linear_velocity,
                "cflt_angular_velocity": angular_velocity,
                "cint_drone_collisions":collisions,
                "cflt_drone_obstacle_distance":obstacle_distances

            })

    def make_tbl_global_state(self):
        start_time = time.time()

        if len(self.output) > 0:
            episode_id = [i for i in range(len(self.output))]
            state_encoding = [";".join(self.output[i]["cstr_global_state"]) for i in range(len(self.output))]
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_global_state duration: {query_duration:.2f} seconds")
            return pd.DataFrame({"cint_episode_id": episode_id, "cstr_state_encoding": str(state_encoding)})

    def make_tbl_drone_actions(self):
        start_time = time.time()

        if len(self.output) > 0:
            episodes = []
            drones = []
            linear_velocity = []
            angular_velocity = []
            for episode in range(len(self.output)):
                for drone in range(self.output[0]["cint_n_drones"]):
                    episodes.append(episode)
                    drones.append(drone)
                    linear_velocity.append(self.output[episode]["actions"][drone][0])
                    angular_velocity.append(self.output[episode]["actions"][drone][1])
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_drone_actions duration: {query_duration:.2f} seconds")
            return pd.DataFrame({
                "cint_episode_id": episodes,
                "cint_drone_id": drones,
                "cflt_linear_velocity": linear_velocity,
                "cflt_angular_velocity": angular_velocity
            })

    def make_tbl_rewards(self):
        start_time = time.time()

        if len(self.output) > 0:
            episodes = [i for i in range(len(self.output))]
            rewards = [self.output[i]["cflt_reward"][0] for i in range(len(self.output))]
            dist = [self.output[i]["cflt_distance_reward"] for i in range(len(self.output))]
            action = [self.output[i]["cflt_action_penalty"] for i in range(len(self.output))]
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_rewards duration: {query_duration:.2f} seconds")
            return pd.DataFrame({"cint_episode_id": episodes,
                                 "cflt_reward": rewards,
                                 "cflt_distance_reward":dist,
                                 "cflt_action_penalty":action})

    def make_tbl_map_data(self):
        start_time = time.time()

        if len(self.map) > 0:
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_map_data duration: {query_duration:.2f} seconds")
            return self.map

    def make_tbl_run_status(self):

        if len(self.run_data) > 0:
            episodes = [self.run_data[i]["episode"] for i in range(len(self.run_data))]
            status = [self.run_data[i]["status"] for i in range(len(self.run_data))]
            time = [self.run_data[i]["time"] for i in range(len(self.run_data))]
            start_time = time.time()
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_run_status duration: %.2f seconds", query_duration)
            return pd.DataFrame({"cint_episode_id": episodes,
                                 "cstr_run_status": status,
                                 "cdtm_status_timestamp": time})

    def generate_tables(self):

        self.tables = {
            "tbl_model_runs": self.make_tbl_model_runs(),
            "tbl_drone_actions": self.make_tbl_drone_actions(),
            "tbl_model_run_params": self.make_tbl_model_run_params(),
            "tbl_rewards": self.make_tbl_rewards(),
            #"tbl_global_state": self.make_tbl_global_state(),
            "tbl_local_state": self.make_tbl_local_state(),
            "tbl_map_data": self.make_tbl_map_data()
            #"tbl_run_status": self.make_tbl_run_status()
        }
        run_id = "999999"
        for name, tbl in self.tables.items():
            tbl.insert(0, "cflt_run_id", run_id)


    def pickle_tables(self):
        with open("utils/pickles/model_output.pkl", "wb") as f:
            pickle.dump(self.tables, f)

        print("Model output pickled")


# -------------------------------------------------------------------------- #
# ------------------------- Helper Functions ------------------------------- #
# -------------------------------------------------------------------------- #
