import time
import pandas as pd
import numpy as np
import pickle
import os

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

        if self.output is not None:
            if len(self.output) > 0:
                for episode in range(len(self.output)):
                    if self.output[episode]["cstr_model_status"] in ["model run initiated", "paused", "stopped"]:
                        run_date = self.output[episode]["cdtm_run_date"]
                        status = self.output[episode]["cstr_model_status"]
                        episode = self.output[episode]["cint_episode"]
                        end_time = time.time()
                        query_duration = end_time - start_time
                        #print(f"make_tbl_model_runs duration: {query_duration:.2f} seconds")
                        return pd.DataFrame({"cdtm_run_date": [run_date],
                                             "cint_episode": [episode],
                                             "cstr_model_status": [status]})

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
            #print(f"make_tbl_model_run_params duration: {query_duration:.2f} seconds")
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
            impr_mult = []
            damage = []
            for episode in range(len(self.output)):
                for drone in range(self.output[0]["cint_n_drones"]):
                    episodes.append(self.output[episode]["cint_episode"])
                    drones.append(drone)
                    x_coords.append(self.output[episode]["local_state"][drone][0])
                    y_coords.append(self.output[episode]["local_state"][drone][1])
                    orientation.append(self.output[episode]["local_state"][drone][2])
                    linear_velocity.append(self.output[episode]["local_state"][drone][3])
                    angular_velocity.append(self.output[episode]["local_state"][drone][4])
                    collisions.append(self.output[episode]["cint_all_collisions"][drone])
                    obstacle_distances.append(self.output[episode]["cflt_drone_obstacle_distance"][drone])
                    impr_mult.append(self.output[episode]["cflt_improvement_multiplier"][drone])
                    damage.append(self.output[episode]["cflt_drone_damage"][drone])

            end_time = time.time()
            query_duration = end_time - start_time
            #print(f"make_tbl_local_state duration: {query_duration:.2f} seconds")
            return pd.DataFrame({
                "cint_episode_id": episodes,
                "cint_drone_id": drones,
                "cflt_x_coord": x_coords,
                "cflt_y_coord": y_coords,
                "cflt_orientation": orientation,
                "cflt_linear_velocity": linear_velocity,
                "cflt_angular_velocity": angular_velocity,
                "cint_drone_collisions":collisions,
                "cflt_drone_obstacle_distance":obstacle_distances,
                "cflt_improvement_multiplier": impr_mult,
                "cflt_drone_damage": damage
            })

    def make_tbl_global_state(self):
        start_time = time.time()
        #print(f"Map Output Length: {len(self.output)}")
        if len(self.output) > 0:
            episode_id = [i for i in range(len(self.output))]
            state_encoding = [";".join(self.output[i]["cstr_global_state"]) for i in range(len(self.output))]
            end_time = time.time()
            query_duration = end_time - start_time
            #print(f"make_tbl_global_state duration: {query_duration:.2f} seconds")
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
            #print(f"make_tbl_drone_actions duration: {query_duration:.2f} seconds")
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
            target_dist = [self.output[i]["cflt_target_distance_reward"] for i in range(len(self.output))]
            direct_dist = [self.output[i]["cflt_direction_reward"] for i in range(len(self.output))]
            dist = [self.output[i]["cflt_distance_reward"] for i in range(len(self.output))]
            action = [self.output[i]["cflt_action_penalty"] for i in range(len(self.output))]
            col_pen = [self.output[i]["cflt_collision_penalty"] for i in range(len(self.output))]
            buf_pen = [self.output[i]["cflt_buffer_penalty"] for i in range(len(self.output))]
            collisions = [sum(self.output[i]["cint_all_collisions"]) for i in range(len(self.output))]
            obstacle_dist = [np.mean(self.output[i]["cflt_drone_obstacle_distance"]) for i in range(len(self.output))]
            improvement = [sum(self.output[i]["cflt_improvement_multiplier"]) for i in range(len(self.output))]
            end_time = time.time()
            query_duration = end_time - start_time
            #print(f"make_tbl_rewards duration: {query_duration:.2f} seconds")
            return pd.DataFrame({"cint_episode_id": episodes,
                                 "cflt_reward": rewards,
                                 "cflt_distance_reward":dist,
                                 "cflt_direction_reward":direct_dist,
                                 "cflt_target_distance_reward": target_dist,
                                 "cflt_action_penalty":action,
                                 "cflt_collision_penalty": col_pen,
                                 "cflt_buffer_penalty": buf_pen,
                                 "cint_all_collisions":collisions,
                                 "cflt_drone_obstacle_distance": obstacle_dist,
                                 "cflt_improvement_multiplier": improvement})


    def make_tbl_map_data(self):
        start_time = time.time()

        if len(self.map_df) > 0:
            end_time = time.time()
            query_duration = end_time - start_time
            #print(f"make_tbl_map_data duration: {query_duration:.2f} seconds")
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
            #print(f"make_tbl_rai: {self.rai_df.columns}")
            return self.rai_df

    def make_tbl_run_status(self):

        if self.run_data is not None:
            if len(self.run_data) > 0:
                episodes = [self.run_data[i]["episode"] for i in range(len(self.run_data))]
                status = [self.run_data[i]["status"] for i in range(len(self.run_data))]
                time = [self.run_data[i]["time"] for i in range(len(self.run_data))]
                start_time = time.time()
                end_time = time.time()
                query_duration = end_time - start_time
                #print(f"make_tbl_run_status duration: %.2f seconds", query_duration)
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
            "tbl_rai": self.make_tbl_rai(),
            "tbl_run_status": self.make_tbl_run_status()
        }

    def pickle_tables(self):
        # File path for the pickle file
        pickle_file_path = "utils/pickles/model_output.pkl"

        # Load existing data if the file exists
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, "rb") as f:
                try:
                    existing_data = pickle.load(f)
                    print("Existing pickle data loaded")
                except Exception as e:
                    print(f"Error loading existing pickle file: {e}")
                    existing_data = {}
        else:
            existing_data = {}

        # Merge existing data with the new data
        for table_name, table_data in self.tables.items():
            if table_name in existing_data:
                # Handle cases where the existing data is None
                if existing_data[table_name] is None:
                    existing_data[table_name] = table_data
                else:
                    # Append new data to the existing table data (if applicable)
                    if table_data is not None:
                        if isinstance(table_data, pd.DataFrame) and isinstance(existing_data[table_name], pd.DataFrame):
                            # Concatenate DataFrames
                            existing_data[table_name] = pd.concat([existing_data[table_name], table_data],
                                                                  ignore_index=True)
                        elif isinstance(table_data, list):
                            # Extend lists
                            existing_data[table_name].extend(table_data)
                        else:
                            # Append other types of data
                            existing_data[table_name].append(table_data)
            else:
                # Add the new table to the existing data
                existing_data[table_name] = table_data

        # Save the updated data back to the pickle file
        with open(pickle_file_path, "wb") as f:
            pickle.dump(existing_data, f)

        print("Model output pickled with merged data")



