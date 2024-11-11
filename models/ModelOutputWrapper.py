import datetime
import gym
import numpy as np
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy
from models.drlss.deep_rl_for_swarms.ma_envs.commons import utils as U
import requests
import time
import os


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

def nearest_obstacle_point(x,y, xy_array):
    xy_array = np.array(xy_array)
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
        self.map_file = map_file
        self.episode_rewards = []
        self.episode = 0
        self.log_data = []
        self.param_data = []
        self.rai_data = pd.DataFrame([self.rai_params])
        self.run_data = []
        self.map_data = map_object
        self.obstacle_df = None
        self.weights_data = []
        self.run_status = "running"
        self.time_limit = 50


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
        return self.env.reset(**kwargs)

    def get_collision_data(self):
        """
        Generates data on collisions for use in reward calculations.

        Uses self.map_data, which is the output of MapPackage. For each object in self.map_data, the distance between
        it and each of the drones is calculated using nearest_obstacle_point(), both with and without a buffer.
        The obstacle distance and damage data is then aggregated at a drone-level.

        Returns:
            self.obstacle_df: Dataframe of collision data.
        """
        self.map_data['obstacles'] = [f"{i}-{j}" for i,j in zip(self.map_data["cstr_obstacle"],self.map_data["cint_obstacle_id"])]
        u_obs = self.map_data['obstacles'].unique()
        obstacle_lookup = {obs: [(x,y) for x,y in zip(self.map_data[self.map_data['obstacles']==obs]['cflt_x_coord'],self.map_data[self.map_data['obstacles']==obs]['cflt_y_coord'])] for obs in u_obs}
        distance_data = []
        for obstacle, coords in obstacle_lookup.items():
            obstacle_risk = self.map_data[self.map_data["cstr_obstacle"]==obstacle]["cflt_obstacle_risk"].unique()
            dists = [nearest_obstacle_point(drone[0], drone[1], coords) for drone in self.world.nodes]
            dists_with_buffer = [nearest_obstacle_point(drone[0], drone[1], coords) + self.rai_params['buffer_zone_size'] for drone in self.world.nodes]
            collisions = [1 if round(dist,0) == 0 else 0 for dist in dists]
            collision_pen = sum([self.rai_params["collision_penalty"] * collision for collision in collisions])
            risks = [obstacle_risk if round(dist,0) == 0 else 0 for dist in dists]
            buffer_penalty = [self.rai_params["buffer_entry_penalty"] if round(dist,0) == 0 else 0 for dist in dists_with_buffer]
            df = pd.DataFrame({
                "cint_drone_id":[i for i, el in enumerate(self.world.nodes)],
                "cstr_obstacle":[obstacle.split("-")[0] for drone in self.world.nodes],
                "cstr_obstacle_id": [obstacle.split("-")[1] for drone in self.world.nodes],
                "cflt_distance_to_obstacle":dists,
                "cflt_distance_to_buffer": dists_with_buffer,
                "cint_collisions":collisions,
                "cint_collision_penalties":collision_pen,
                "cint_risk": risks,
                "cint_buffer_penalty": buffer_penalty
            })
            distance_data.append(df)
        dfs = pd.concat(distance_data)
        dfs["cflt_distance_to_obstacle"] = pd.to_numeric(dfs["cflt_distance_to_obstacle"], errors="coerce")
        dfs["cflt_distance_to_buffer"] = pd.to_numeric(dfs["cflt_distance_to_obstacle"], errors="coerce")
        dfs["cint_collisions"] = pd.to_numeric(dfs["cint_collisions"], errors="coerce")
        dfs["cint_collision_penalties"] = pd.to_numeric(dfs["cint_collision_penalties"], errors="coerce")
        dfs["cint_risk"] = pd.to_numeric(dfs["cint_risk"], errors="coerce")
        dfs["cint_buffer_penalty"] = pd.to_numeric(dfs["cint_buffer_penalty"], errors="coerce")
        dfs = dfs.fillna(0)

        self.obstacle_df = dfs.groupby("cint_drone_id", as_index=False).agg(
            {"cflt_distance_to_obstacle":"min",
             "cint_collisions":"sum",
             "cint_risk":"sum",
             "cint_collision_penalties": "sum",
             "cint_buffer_penalty":"sum"}
        )
        self.map_data = self.map_data.drop('obstacles', axis=1)

    def get_rai_reward(self, actions):
        self.get_collision_data()
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)

        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        # All possible RAI penalties
        ## Avoid Basic Collision
        collision_pen = self.obstacle_df["cint_collision_penalties"].to_numpy()
        obs_specific_penalty = self.obstacle_df["cint_risk"].to_numpy()

        ## Avoid Buffer Zone
        all_buffer_penalties = self.obstacle_df["cint_buffer_penalty"].to_numpy()

        if self.rai_params["avoid_collisions"]==True:

            r = - dist_rew - action_pen - collision_pen - obs_specific_penalty
            r = np.ones((self.nr_agents,)) * r
        elif self.rai_params["avoid_buffer_zones"] == True:
            r = - dist_rew - action_pen - collision_pen - obs_specific_penalty - all_buffer_penalties
            r = np.ones((self.nr_agents,)) * r
        else:
            r = - dist_rew - action_pen
            r = np.ones((self.nr_agents,)) * r

        return r

    def rai_reward_data(self, actions):
        self.get_collision_data()
        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius

        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions ** 2)

        # All possible RAI penalties
        all_collisions = self.obstacle_df["cint_collisions"].to_numpy()
        all_obstacle_distances = self.obstacle_df["cflt_distance_to_obstacle"].to_numpy()
        all_obstacles_cap = np.where(all_obstacle_distances > self.comm_radius, self.comm_radius,
                                     all_obstacle_distances)
        ## Avoid Basic Collision
        collision_pen = self.obstacle_df["cint_collision_penalties"].to_numpy()
        obs_specific_penalty = self.obstacle_df["cint_risk"].to_numpy()
        r_collisions = - dist_rew - action_pen - collision_pen - obs_specific_penalty

        ## Avoid Buffer Zone
        all_buffer_penalties = self.obstacle_df["cint_buffer_penalty"].to_numpy()
        r_collision_buffer = - dist_rew - action_pen - collision_pen - obs_specific_penalty - all_buffer_penalties

        rai_data = {
            "cflt_distance_reward": dist_rew,
            "cflt_action_penalty": action_pen,
            "cflt_reward_with_collisions": r_collisions,
            "cflt_reward_with_collisions_buffer": r_collision_buffer,
            "cint_all_collisions": all_collisions,
            "cflt_capped_obstacle_distance": all_obstacles_cap,
            "cflt_obstacle_distance": all_obstacle_distances
        }
        return rai_data


    def step(self, action):
        state = self.env.state if hasattr(self.env, "state") else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        reward = self.get_rai_reward(action)
        rai_reward = self.rai_reward_data(action)

        check_status()

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
            "cflt_drone_obstacle_distance": rai_reward["cflt_obstacle_distance"]
        }

        self.log_data.append(log_entry)
        self.episode_rewards.append(reward)
        #self.run_data.append(run_entry)
        self.episode += 1
        update_episode(self.episode)
        print(f'On episode: {self.episode}', flush=True)
        if check_status():
            print(f"Model paused at episode {self.episode}.", flush=True)
            while check_status():
                time.sleep(1)  # Wait until pause flag is cleared
            print("Model resumed.", flush=True)

        # Batch output for mid-run database commits
        if self.episode == 1:
            self.run_parameters()

        if self.episode % 20 == 0:
            if self.episode == 20:
                self.batched_commits()
            else:
                self.batched_commits(first=False)
            update_db("commit")

        time.sleep(0.25)
        return next_state, reward, done, info

    def close(self):
        # Commit any remaining data
        output = OutputObject(self.log_data,
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
            out = OutputObject(self.log_data, self.param_data, self.map_data, self.rai_data)
            out.generate_tables()
            out.pickle_tables()
            self.log_data = []
        else:
            out = OutputObject(self.log_data, [], [], [])
            out.generate_tables()
            out.pickle_tables()
            self.log_data = []
        print("Batch is pickled")



# -------------------------------------------------------------------------- #
# -------------- OutputObject: converts output to DataFrames --------------- #
# -------------------------------------------------------------------------- #
class OutputObject:
    def __init__(self, output, params, map, rai, run_data = None, output_type="json"):
        self.output = output
        self.params = params[0] if len(params) > 0 else []
        self.run_data = run_data
        self.map_df = map
        self.rai_df = rai
        self.output_type = output_type
        self.tables = {}


    def make_tbl_model_runs(self):
        start_time = time.time()

        if len(self.output) > 0:
            run_date = self.output[0]["cdtm_run_date"]
            terminal_episode = [self.output[i]["cbln_terminal"] for i in range(len(self.output))].index(False)
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
                    episodes.append(self.output[episode]["cint_episode"])
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
        print(f"Map Output Length: {len(self.output)}")
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
                    episodes.append(self.output[episode]["cint_episode"])
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
            episodes = [self.output[i]["cint_episode"] for i in range(len(self.output))]
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

        if len(self.map_df) > 0:
            end_time = time.time()
            query_duration = end_time - start_time
            print(f"make_tbl_map_data duration: {query_duration:.2f} seconds")
            return self.map_df

    def make_tbl_rai(self):
        start_time = time.time()

        if len(self.rai_df) > 0:
            end_time = time.time()
            query_duration = end_time - start_time
            col_names = ["cbln_avoid_collisions","cflt_collision_penalty", "cbln_avoid_buffer_zones",
             "cflt_buffer_zone_size", "cflt_buffer_entry_penalty", "cint_expected_completion_time",
             "cflt_swarm_damage_tolerance","cflt_drone_damage_tolerance"]
            self.rai_df.columns = col_names
            print(f"make_tbl_map_data duration: {query_duration:.2f} seconds")
            return self.rai_df

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
            "tbl_map_data": self.make_tbl_map_data(),
            "tbl_rai": self.make_tbl_rai()
            #"tbl_run_status": self.make_tbl_run_status()
        }

    def pickle_tables(self):
        with open("utils/pickles/model_output.pkl", "wb") as f:
            pickle.dump(self.tables, f)

        print("Model output pickled")

