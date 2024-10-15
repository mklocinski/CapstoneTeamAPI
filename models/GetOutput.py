import json
import datetime
import gym
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------- #
# ---------------------------- Description --------------------------------- #
# -------------------------------------------------------------------------- #
# This script contains two classes,
# > OutputWrapper:
#   > Input: DRL models
#   > Output: JSON/dictionaries of DRL output.
#   > Description: This is an edited version of the output wrapper we've been using.
#       All extraneous or redundant information has been removed.
# > OutputToDict:
#   > Input: JSON/dictionaries created by DRLOutput
#   > Output: Dictionary of tables corresponding to the tables defined in
#       'DRL Data Dictionary.xlsx'; can also bulk append output to database.
#   > Description: Flattens the JSON from DRLOutput into DataFrames that will be
#       added to the API database (\app\data_models.py)


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
                 log_file='output.json',
                 param_file='param.json'
                 ):
        super(OutputWrapper, self).__init__(env)
        self.run_date = datetime.datetime.now().strftime('%Y%m%d_%H%M_%S')
        self.env_type = env_type
        self.environment_params= environment_params
        self.model_params= model_params
        self.log_file = log_file
        self.param_file = param_file
        self.episode_rewards = []
        self.episode = 0
        self.log_data = []
        self.param_data =[]


    def run_parameters(self, **kwargs):
        env_entry = {}
        for de, val in self.environment_params.items():
            env_entry[de] = val

        model_entry = {}
        for de, val in self.model_params.items():
            model_entry[de] = val

        param_entry = {'environment_id':self.env_type,
                       **env_entry, **model_entry}

        self.param_data.append(param_entry)

    def reset(self, **kwargs):
        """Resets the environment and initializes logging for a new episode."""
        self.episode_rewards = []
        self.episode = 0
        return self.env.reset(**kwargs)

    # Primary function, extracts output
    def step(self, action):
        """Logs state, action, reward, next state, and done."""
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        log_entry = {
            "n_drones": self.env.nr_agents,
            "step": self.episode,
            "run_date": self.run_date,
            "terminal": done,
            "global_state": state,
            "local_state": info["state"],
            "actions": info["actions"],
            "reward": reward
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
        print(serializable_param_data)
        serializable_log_data = self.convert_ndarray(self.log_data)

        print("Writing param data to", self.param_file)
        with open(self.param_file, 'w') as f:
            json.dump(serializable_param_data, f, indent=4)

        print("Writing log data to", self.log_file)
        with open(self.log_file, 'w') as f:
            json.dump(serializable_log_data, f, indent=4)

        # Close the environment
        super(OutputWrapper, self).close()

# -------------------------------------------------------------------------- #
# ------------- ModelOutput: converts output to DataFrames --------------- #
# -------------------------------------------------------------------------- #
# All functions except the last generate a specific table as documented
# in 'DRL Data Dictionary.xlsx.'
# The final function adds a primary key, 'run_id', to each table

class ModelOutput:
    def __init__(self, output, params, output_type='json'):
        self.output = output
        self.params = params
        self.output_type = output_type
        self.tables = {}


    def make_tbl_model_runs(self):
        run_date = self.output[0]["run_date"]
        terminal_episode = [self.output[i]["terminal"] for i in range(len(self.output))].index(True)

        return(pd.DataFrame({'run_date': [run_date],
                             'terminal_episode': [terminal_episode]}))

    def make_tbl_model_run_params(self):

        return pd.DataFrame(self.params, index=[0])

    def make_tbl_local_state(self):
        episodes = []
        drones = []
        x_coords = []
        y_coords = []
        orientation = []
        linear_velocity = []
        angular_velocity = []

        for episode in range(len(self.output)):
            es = [episode for i in range(self.output[0]["n_drones"])]
            episodes.append(es)
            ds = [i+1 for i in range(self.output[0]["n_drones"])]
            drones.append(ds)
            xs = self.output[episode]["local_state"][0]
            x_coords.append(xs)
            ys = self.output[episode]["local_state"][1]
            y_coords.append(ys)
            os = self.output[episode]["local_state"][2]
            orientation.append(os)
            lvs = self.output[episode]["local_state"][3]
            linear_velocity.append(lvs)
            avs = self.output[episode]["local_state"][4]
            angular_velocity.append(avs)

        return(pd.DataFrame({
            'episode_id':episodes,
            'drone_id': drones,
            'x_coord': x_coords,
            'y_coord': y_coords,
            'orientation': orientation,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity
        }))

    def make_tbl_global_state(self):
        episode_id = [i for i in range(len(self.output))]
        state_encoding = [self.output[i]["global_state"] for i in range(len(self.output))]

        return(pd.DataFrame({
            'episode_id': episode_id,
            'state_encoding': state_encoding
        }))

    def make_tbl_drone_actions(self):
        episodes = []
        drones = []
        linear_velocity = []
        angular_velocity = []

        for episode in range(len(self.output)):
            es = [episode for i in range(self.output[0]["n_drones"])]
            episodes.append(es)
            ds = [i+1 for i in range(self.output[0]["n_drones"])]
            drones.append(ds)
            lvs = [i[0] for i in self.output[episode]['actions']]
            linear_velocity.append(lvs)
            avs = [i[1] for i in self.output[episode]['actions']]
            angular_velocity.append(avs)

        return(pd.DataFrame({
            'episode_id': episodes,
            'drone_id': drones,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity
        }))

    def make_tbl_rewards(self):
        episodes = [i for i in range(len(self.output))]
        rewards = [self.output[i]['reward'][0] for i in range(len(self.output))]

        return(pd.DataFrame({
            'episode_id': episodes,
            'reward': rewards
        }))

    def generate_tables(self):
        # run id is the timestamp for the models's run
        run_id = round(datetime.datetime.now().timestamp(),0)
        self.tables = {'tbl_model_runs': self.make_tbl_model_runs(),
                       'tbl_model_run_params': self.make_tbl_model_run_params(),
                    'tbl_local_state': self.make_tbl_local_state(),
                    'tbl_global_state': self.make_tbl_global_state(),
                    'tbl_drone_actions': self.make_tbl_drone_actions(),
                    'tbl_rewards': self.make_tbl_rewards()}

        for name, tbl in self.tables.items():
            tbl.insert(0, 'run_id', run_id)

