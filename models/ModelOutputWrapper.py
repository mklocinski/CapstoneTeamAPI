import json
import datetime
import gym
import numpy as np
import pandas as pd
import pickle
import json


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

# -------------------------------------------------------------------------- #
# ------------- OutputWrapper(ModelObject): models output wrapper --------------- #
# -------------------------------------------------------------------------- #
# Edited version of original DRL output wrapper
class OutputWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 env_type,
                 environment_params,
                 model_params,
                 map_object,
                 log_file="output.json",
                 param_file="param.json",
                 map_file="map.json"
                 ):
        super(OutputWrapper, self).__init__(env)
        self.run_date = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S")
        self.env_type = env_type
        self.environment_params= environment_params
        self.model_params= model_params
        self.log_file = log_file
        self.param_file = param_file
        self.map_file = map_file
        self.episode_rewards = []
        self.episode = 0
        self.log_data = []
        self.param_data =[]
        map_data_df = map_object.map_all_coordinates_df.reset_index(drop=True)
        self.map_data = map_data_df.to_json(orient='index')

    def run_parameters(self, **kwargs):
        env_entry = {}
        for de, val in self.environment_params.items():
            env_entry[de] = val

        model_entry = {}
        for de, val in self.model_params.items():
            model_entry[de] = val

        param_entry = {"environment_id":self.env_type,
                       **env_entry, **model_entry}

        self.param_data.append(param_entry)

    def reset(self, **kwargs):
        """Resets the environment and initializes logging for a new episode."""
        self.episode_rewards = []
        self.episode = 0
        return self.env.reset(**kwargs)

    # Primary function, extracts output
    def step(self, action):
        state = self.env.state if hasattr(self.env, "state") else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        log_entry = {
            "cint_n_drones": self.env.nr_agents,
            "cint_episode": self.episode,
            "cdtm_run_date": self.run_date,
            "cbln_terminal": done,
            "cstr_global_state": state,
            "local_state": info["state"],
            "actions": info["actions"],
            "cflt_reward": reward
        }

        # Append log entry to the list
        self.log_data.append(log_entry)

        self.episode += 1
        self.episode_rewards.append(reward)

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

        # #print("Writing param data to", self.param_file)
        # with open(self.param_file, "w") as f:
        #     json.dump(serializable_param_data, f, indent=4)
        #
        # #print("Writing log data to", self.log_file)
        # with open(self.log_file, "w") as f:
        #     json.dump(serializable_log_data, f, indent=4)
        #
        # with open(self.map_file, "w") as f:
        #      json.dump(self.map_data)

        # Close the environment
        super(OutputWrapper, self).close()

    def batched_commits(self, n=100):
        if len(self.param_data)==n:
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
# All functions except the last generate a specific table as documented
# in "DRL Data Dictionary.xlsx."
# The final function adds a primary key, "run_id", to each table

class OutputObject:
    def __init__(self, output, params, map, output_type="json"):
        self.output = output
        self.params = params[0]
        self.map = map
        self.output_type = output_type
        self.tables = {}


    def make_tbl_model_runs(self):
        print("...start making tbl_model_runs...")
        if len(self.output) > 0:
            run_date = self.output[0]["cdtm_run_date"]
            terminal_episode = [self.output[i]["cbln_terminal"] for i in range(len(self.output))].index(True)
            print("...end making tbl_model_runs...")
            return(pd.DataFrame({"cdtm_run_date": [run_date],
                                 "cbln_terminal_episode": [terminal_episode]}))

    def make_tbl_model_run_params(self):
        print("...start making tbl_model_run_params...")
        if len(self.params) > 0:
            ps = {'cstr_environment_id':self.params['environment_id'],
                'cint_nr_agents':self.params['nr_agents'],
                 'cstr_obs_mode':self.params['obs_mode'],
                 'cint_comm_radius':self.params['comm_radius'],
                 'cint_world_size':self.params['world_size'],
                 'cint_distance_bins':self.params['distance_bins'],
                 'cint_bearing_bins':self.params['bearing_bins'],
                 'cbln_torus':self.params['torus'],
                 'cstr_dynamics':self.params['dynamics'],
                 'cint_timesteps_per_batch':self.params['timesteps_per_batch'],
                 'cflt_max_kl':self.params['max_kl'],
                'cint_cg_iters':self.params['cg_iters'],
                 'cflt_cg_damping':self.params['cg_damping'],
                 'cflt_gamma':self.params['gamma'],
                 'cflt_lam':self.params['lam'],
                 'cint_vf_iters':self.params['vf_iters'],
                 'cint_vf_stepsize':self.params['vf_stepsize']}
            print("...end making tbl_model_run_params...")
            return pd.DataFrame(ps, index=[0])

    def make_tbl_local_state(self):
        if len(self.output)>0:
            episodes = []
            drones = []
            x_coords = []
            y_coords = []
            orientation = []
            linear_velocity = []
            angular_velocity = []
            print("...start making tbl_local_state...")
            for episode in range(len(self.output)):
                for drone in range(self.output[0]["cint_n_drones"]):
                    episodes.append(episode)
                    drones.append(drone)
                    x_coords.append(self.output[episode]["local_state"][drone][0])
                    y_coords.append(self.output[episode]["local_state"][drone][1])
                    orientation.append(self.output[episode]["local_state"][drone][2])
                    linear_velocity.append(self.output[episode]["local_state"][drone][3])
                    angular_velocity.append(self.output[episode]["local_state"][drone][4])

            print("...end making tbl_local_state...")
            return(pd.DataFrame({
                "cint_episode_id": episodes,
                "cint_drone_id": drones,
                "cflt_x_coord": x_coords,
                "cflt_y_coord": y_coords,
                "cflt_orientation": orientation,
                "cflt_linear_velocity": linear_velocity,
                "cflt_angular_velocity": angular_velocity
        }))

    def make_tbl_global_state(self):
        if len(self.output)>0:
            print("...start making tbl_global_state...")
            episode_id = [i for i in range(len(self.output))]
            #state_encoding = [str(','.join(self.output[i]["cstr_global_state"])) for i in range(len(self.output))]
            state_encoding = ["test" for i in range(len(self.output))]
            print("...end making tbl_global_state...")
            return(pd.DataFrame({
                "cint_episode_id": episode_id,
                "cstr_state_encoding": state_encoding
            }))

    def make_tbl_drone_actions(self):
        print("...start making tbl_drone_actions...")
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
            print("...end making tbl_drone_actions...")

            # with open("debug.pkl", "wb") as f:
            #     pickle.dump(drones, f)

            return pd.DataFrame({
                "cint_episode_id": episodes,
                "cint_drone_id": drones,
                "cflt_linear_velocity":linear_velocity,
                "cflt_angular_velocity": angular_velocity
            })

    def make_tbl_rewards(self):
        print("...start making tbl_rewards...")
        if len(self.output) > 0:
            episodes = [i for i in range(len(self.output))]
            rewards = [self.output[i]["cflt_reward"][0] for i in range(len(self.output))]
            print("...end making tbl_rewards...")
            return(pd.DataFrame({
                "cint_episode_id": episodes,
                "cflt_reward": rewards
            }))

    def make_tbl_map_data(self):
        print("...start making tbl_map_data...")
        if len(self.map) > 0:
            x_coord = []
            y_coord = []
            obstacle = []
            for key, item in json.loads(self.map).items():
                x_coord.append(item["x_coord"])
                y_coord.append(item["y_coord"])
                obstacle.append(item["obstacle"])
            print("...end making tbl_map_data...")
            return pd.DataFrame({
                "cflt_x_coord": x_coord,
                "cflt_y_coord": y_coord,
                "cint_obstacle": obstacle
            })

    def generate_tables(self):
        # run id is the timestamp for the models"s run
        run_id = round(100000*datetime.datetime.now().timestamp(),0)
        self.tables = {"tbl_model_runs": self.make_tbl_model_runs(),
                       "tbl_drone_actions": self.make_tbl_drone_actions(),
                       "tbl_model_run_params": self.make_tbl_model_run_params(),
                       "tbl_rewards": self.make_tbl_rewards(),
                    "tbl_global_state": self.make_tbl_global_state(),
                    "tbl_local_state": self.make_tbl_local_state(),
                       "tbl_map_data": self.make_tbl_map_data()
                       }

        for name, tbl in self.tables.items():
            tbl.insert(0, "cflt_run_id", run_id)
            #tbl.reset_index(drop=True)

        print(self.tables.keys())

    def pickle_tables(self):

        with open("model_output.pkl", "wb") as f:
            pickle.dump(self.tables, f)

        print("Model output pickled")

    #def debug_pickle(self):

        #with open("debug.pkl", "wb") as f:
            #pickle.dump(self.output[0], f)

# -------------------------------------------------------------------------- #
# ------------------------- Helper Functions ------------------------------- #
# -------------------------------------------------------------------------- #

def scalarize(pickled_table):
    tbl = {}
    for col in pickled_table.columns:
        if col[:5] == 'cint_':
            tbl[col] = [int(i) for i in pickled_table[col]]
        elif col[:5] == 'cdtm_':
            tbl[col] = [datetime.datetime.strptime(i, "%Y%m%d_%H%M_%S") for i in pickled_table[col]]
        elif col[:5] == 'cflt_':
            tbl[col] = [float(i) for i in pickled_table[col]]
        elif col[:5] == 'cbln_':
            tbl[col] = [bool(i) for i in pickled_table[col]]
        elif col[:5] == 'cstr_':
            tbl[col] = [str(i) for i in pickled_table[col]]
        else:
            tbl[col] = [str(i) for i in pickled_table[col]]
    return tbl


