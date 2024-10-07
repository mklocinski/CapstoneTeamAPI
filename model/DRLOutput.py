import json
import datetime
import pandas as pd


# -------------------------------------------------------------------------- #
# ---------------------------- Description --------------------------------- #
# -------------------------------------------------------------------------- #
# This script contains two classes,
# > DRLOutput:
#   > Input: DRL model
#   > Output: JSON of DRL output.
#   > Description: This is an edited version of the output wrapper we've been using.
#       All extraneous or redundant information has been removed.
# > FlattenOutput:
#   > Input: JSON created by DRLOutput
#   > Output: Dictionary of tables corresponding to the tables defined in
#       'DRL Data Dictionary.xlsx'.
#   > Description: Flattens the JSON from DRLOutput into DataFrames that will be
#       added to the API database (\app\data_models.py)


# -------------------------------------------------------------------------- #
# ------------- DRLOutput(ModelObject): model output wrapper --------------- #
# -------------------------------------------------------------------------- #
# Edited version of original DRL output wrapper
class DRLOutput(ModelObject):
    def __init__(self, env, log_file = 'output.json', param_file = 'param.json'):
        super(DRLOutput, self).__init__(env)
        self.run_date = datetime.datetime.now()
        self.log_file = log_file
        self.param_file = param_file
        self.agents = self.env.nr_agents
        self.episode_rewards = []
        self.episode = 0
        self.log_data = []

    def reset(self, **kwargs):
        """Resets the environment and initializes logging for a new episode."""
        self.episode_rewards = []
        self.episode = 0

    # Primary function, extracts output
    def step(self, action):
        """Logs state, action, reward, next state, and done."""
        state = self.env.state if hasattr(self.env, 'state') else self.env.observation_space.sample()
        next_state, reward, done, info = self.env.step(action)

        log_entry = {
            "n_drones": self.agents,
            "step": self.episode,
            "run_date": self.run_date,
            "terminal": info["done"],
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


    def close(self):
        # Convert the log data to be JSON serializable
        serializable_log_data = self.convert_ndarray(self.log_data)

        # Print for debugging
        print("Writing log data to", self.log_file)

        # Write log data to the log file
        with open(self.log_file, 'w') as f:
            json.dump(serializable_log_data, f, indent=4)

        # Close the environment
        super(DRLOutput, self).close()

# -------------------------------------------------------------------------- #
# ------------- FlattenOutput: converts output to DataFrames --------------- #
# -------------------------------------------------------------------------- #
# All functions except the last generate a specific table as documented
# in 'DRL Data Dictionary.xlsx.'
# The final function adds a primary key, 'run_id', to each table

class FlattenOutput:
    def __init__(self, output, output_type='json'):
        self.output = output
        self.output_type = output_type

    def make_tbl_model_run(self):
        run_date = self.output[0]["run_date"]
        terminal_episode = [i["terminal"] for i in range(len(self.output))].index(True)

        return(pd.DataFrame({'run_date': run_date,
                             'terminal_episode': terminal_episode}))

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
            xs = self.output[episode]["state"][0]
            x_coords.append(xs)
            ys = self.output[episode]["state"][1]
            y_coords.append(ys)
            os = self.output[episode]["state"][2]
            orientation.append(os)
            lvs = self.output[episode]["state"][3]
            linear_velocity.append(lvs)
            avs = self.output[episode]["state"][4]
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
        # run id is the timestamp for the model's run
        run_id = round(datetime.datetime.now().timestamp(),0)
        tables = {'tbl_model_runs': self.make_tbl_model_runs(),
                    'tbl_local_state': self.make_tbl_local_state(),
                    'tbl_global_state': self.make_tbl_global_state(),
                    'tbl_drone_actions': self.make_tbl_drone_actions(),
                    'tbl_rewards': self.make_tbl_rewards()}

        for name, tbl in tables.items():
            tbl.insert(0, 'run_id', run_id)

        return tables

