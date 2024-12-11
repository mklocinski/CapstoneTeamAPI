from flask import Blueprint, jsonify, request, current_app
from utils.database_utils import tbl_utilities, tbl_batches, scalarize
import subprocess
import os
import pickle
import pandas as pd
from sqlalchemy import text, func
import json
import warnings
from .data_models import tbl_model_runs, tbl_local_state, tbl_rewards, tbl_global_state, tbl_drone_actions, tbl_model_run_params, tbl_map_data, tbl_rai
from . import db
import datetime
import platform
import traceback
import tempfile
import time
import threading
import logging
import queue


# Set-up
main = Blueprint('main', __name__)
warnings.filterwarnings("ignore")

# Model Variables
model_status = {"status": "idle", "episode": 0, "run_id":0}
model_thread = None
stop_flag = threading.Event()
logging.basicConfig(level=logging.INFO)
thread_logger = logging.getLogger("model_thread")
completion_queue = queue.Queue()

# Flag Variables
db_status_path = "/CapstoneTeamAPI/utils/status/data_commit.txt"
episode_path = "/CapstoneTeamAPI/utils/status/model_episode.txt"
status_path = "/CapstoneTeamAPI/utils/status/model_status.txt"
model_output_path = "/CapstoneTeamAPI/utils/pickles/model_output.pkl"
backup_path = "/CapstoneTeamAPI/utils/pickles/uncommitted_output.pkl"

@main.route('/')
def home():
    setup_flag_files()
    #cleanup_committed_episodes()
    return jsonify({"message": "Welcome to the XRAI API. Please refer to the documentation for available endpoints."}), 200

# def make_environment_map(map_size,no_fly_zones,humans,buildings,trees,animals,fires,
#                          no_fly_zones_damage,humans_damage,buildings_damage,trees_damage,animals_damage, fire_damage):
def make_environment_map(world, map_params):
    current_app.logger.info(">> Generating map data")
    #map_params["map_size"] = [int(world),int(world)]
    current_app.logger.info(map_params)
    import models.ObstacleGenerator as m
    obs_map = m.EnvironmentMap(**map_params)
    current_app.logger.info(">> >> Map object created")
    obs_map.generate_obstacle_data()
    map_data = obs_map.dataframe.reset_index(drop=True).to_json()

    current_app.logger.info(">> Map data generated")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
        temp_file.write(map_data)
        temp_file_path = temp_file.name
        current_app.logger.info(f">> Map data written to temporary file: {temp_file_path}")

    return temp_file_path

def setup_flag_files():
    if os.path.exists(model_output_path):
        os.remove(model_output_path)
    if not os.path.exists(status_path):
        with open(status_path, 'w') as file:
            file.write("idle")
        print(f"{status_path} created and flag set to 'idle'.")
    else:
        with open(status_path, 'w') as file:
            file.write("idle")
        print(f"{status_path} created and flag set to 'idle'.")

    if not os.path.exists(db_status_path):
        with open(db_status_path, 'w') as file:
            file.write("queuing")
        print(f"{db_status_path} created and flag set to 'queuing'.")
    else:
        with open(db_status_path, 'w') as file:
            file.write("queuing")
        print(f"{db_status_path} created and flag set to 'queuing'.")

    if not os.path.exists(episode_path):
        with open(episode_path, 'w') as file:
            file.write("0")
        print(f"{episode_path} created and flag set to '0'.")
    else:
        with open(episode_path, 'w') as file:
            file.write("0")
        print(f"{episode_path} created and flag set to '0'.")

def initialize_committed_episodes():
    # Initialize an empty set
    with open('committed_episodes.pkl', 'wb') as file:
        pickle.dump(set(), file)
    current_app.logger.info("Initialized committed_episodes.pkl")

# def cleanup_committed_episodes():
#     if os.path.exists('committed_episodes.pkl'):
#         os.remove('committed_episodes.pkl')
#         current_app.logger.info("Deleted committed_episodes.pkl after model completion.")


def set_model_status(status):
    if not os.path.exists(status_path):
        with open(status_path, 'w') as file:
            file.write(status)
        print(f"{status_path} flag set to '{status}'.")
    else:
        with open(status_path, 'w') as file:
            file.write(status)
        print(f"{status_path} flag set to '{status}'.")

def batch_insertion(run_id, local_session):
    current_app.logger.info('>> Batch commit to database')

    # Check if the pickle file exists
    if not os.path.exists(model_output_path):
        error_message = f"Model output file {model_output_path} not found."
        current_app.logger.error(error_message)
        return False, error_message

    # Check if the pickle file is empty
    if os.path.getsize(model_output_path) == 0:
        error_message = f"Model output file {model_output_path} is empty."
        current_app.logger.error(error_message)
        return False, error_message

    # Load the pickle file
    try:
        with open(model_output_path, 'rb') as f:
            model_output = pickle.load(f)
    except Exception as e:
        error_message = f"Failed to load pickle file {model_output_path}: {e}"
        current_app.logger.error(error_message)
        return False, error_message

    # Ensure the pickle file contains data
    if not model_output:
        error_message = f"Model output in {model_output_path} is empty after loading."
        current_app.logger.error(error_message)
        return False, error_message

    # Process the model output and commit to the database
    try:
        for name, tbl in model_output.items():
            if tbl is not None:
                tbl.insert(0, "cflt_run_id", run_id)

        for key, val in model_output.items():
            if val is not None:
                scalarized = pd.DataFrame(scalarize(val))
                if key == "tbl_rai":
                    current_app.logger.info(">> RAI table")
                    for i in val.keys():
                        current_app.logger.info(val[i])

                if scalarized.empty:
                    current_app.logger.warning(f"Scalarized data for {key} is empty. Skipping...")
                    continue

                batch = []
                for ix, row in scalarized.iterrows():
                    one_row = {col: row[col] for col in scalarized.columns}


                    if key in tbl_utilities:
                        tbl_row = tbl_utilities[key](**one_row)
                        batch.append(tbl_row)

                if not batch:
                    current_app.logger.warning(f"No rows to commit for {key}. Skipping...")
                    continue

                local_session.bulk_save_objects(batch)
                local_session.commit()
                current_app.logger.info(f"Batch committed to {key}")

        local_session.close()
        current_app.logger.info("Model output committed to the database successfully.")

        # Remove the pickle file after successful commit
        os.remove(model_output_path)
        current_app.logger.info(f"Pickle file {model_output_path} cleared after successful commit.")

        with open(db_status_path, 'w') as file:
            file.write("queuing")

        return True, "Model ran and data committed successfully."

    except Exception as e:
        local_session.rollback()
        error_message = f"An error occurred during database commit: {e}"
        traceback_info = traceback.format_exc()
        current_app.logger.error(f"{error_message}\nTraceback:\n{traceback_info}")
        return False, f"{error_message}\nTraceback:\n{traceback_info}"



# Update only model run table at model stop
def update_single_table(table, values):
    try:
        # Create a new row object
        row = table(**values)

        # Add and commit the row to the database
        db.session.add(row)
        db.session.commit()

        current_app.logger.info(f"Successfully inserted into {table.__tablename__}: {values}")
        return True, f"Row inserted into {table.__tablename__} successfully."
    except Exception as e:
        db.session.rollback()
        error_message = f"Error inserting into {table.__tablename__}: {e}"
        current_app.logger.error(error_message)
        return False, error_message


@main.route('/model/standard/run_xrai', methods=['POST'])
def post_standard_run_xrai_system():
    current_app.logger.info("Starting standard model run:")
    setup_flag_files()
    initialize_committed_episodes()
    set_model_status("initializing")
    try:
        current_app.logger.info(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        global model_status
        model_status["status"] = "running"

        # Get run parameters
        data = request.get_json()
        environ_params = json.dumps(data.get("environment_parameters"))
        model_params = json.dumps(data.get("model_parameters"))
        map_params = json.dumps(data.get("map_parameters"))
        rai_params = json.dumps(data.get("rai_parameters"))
        current_app.logger.info(">> Parameters received")

        # Make environment map
        try:
            world_size=json.loads(environ_params)
            map_params = json.loads(map_params)
            current_app.logger.info(map_params)
            map_path = make_environment_map(world_size.get("world_size"), map_params)

            current_app.logger.info("make_environment_map completed")
        except Exception as e:
            current_app.logger.info(f"Error calling make_environment_map: {e}")


        # Create run_id
        run_id = round(1000 * datetime.datetime.now().timestamp(), 0)
        model_status["run_id"] = run_id
        current_app.logger.info(model_status["run_id"])

        # Run model asynchronously
        set_model_status("running")
        current_app.logger.info(f"Map path: {map_path}")
        start_model_thread(map_path, environ_params, model_params, rai_params, run_id=model_status["run_id"])

        return jsonify({'model': 'Successful run'}), 200
    except Exception as e:

        return jsonify({'model': str(e)}), 500


@main.route('/model/no_app/run_xrai', methods=['POST'])
def post_no_app_run_xrai():
    current_app.logger.info("Starting model run:")
    setup_flag_files()
    initialize_committed_episodes()
    set_model_status("initializing")

    try:
        current_app.logger.info(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        global model_status
        model_status["status"] = "running"

        # Default parameters
        default_environment_params = {"environment_id": "Rendezvous",
                                      "nr_agents": 20,
                                      "obs_mode": "sum_obs_acc",
                                      "comm_radius": 2,
                                      "world_size": 100,
                                      "distance_bins": 8,
                                      "bearing_bins": 8,
                                      "torus": False,
                                      "dynamics": "unicycle_acc"}
        default_model_params = {'timesteps_per_batch': 10,
                                'max_kl': 0.02,
                                'cg_iters': 10,
                                'cg_damping': 0.1,
                                'gamma': 0.95,
                                'lam': 0.95,
                                'vf_iters': 5,
                                'vf_stepsize': 0.001},
        default_map_params = {'target': {'target_x_coordinate': 20, 'target_y_coordinate': 20},
                              'humans': {'count': 1, 'damage': 10, 'sizes': [3], 'random': 1, 'positions': ['None set']},
                              'no_fly_zones': {'count': 1, 'damage': 10, 'sizes': [3], 'random': 1, 'positions': ['None set']},
                              'buildings': {'count': 1, 'damage': 10, 'sizes': [3], 'random': 1, 'positions': ['None set']},
                              'trees': {'count': 1, 'damage': 10, 'sizes': [3], 'random': 1, 'positions': ['None set']},
                              'animals': {'count': 1, 'damage': 10, 'sizes': [3], 'random': 1, 'positions': ['None set']},
                              'fires': {'count': 1, 'damage': 10, 'sizes': [3], 'random': 1, 'positions': ['None set']}}

        default_rai_params =  {'expected_completion_time': 10,
                               'swarm_damage_tolerance': 10,
                               'individual_drone_damage_tolerance': 10,
                               'avoid_collisions': 1,
                               'avoid_buffer_zones': False,
                               'buffer_zone_size': False,
                               'collision_penalty': 10,
                               'buffer_entry_penalty': 10}

        # Merge user-provided parameters with defaults
        data = request.get_json() or {}
        environ_params = {**default_environment_params, **(data.get("environment_parameters") or {})}
        model_params = {**default_model_params, **(data.get("model_parameters") or {})}
        map_params = {**default_map_params, **(data.get("map_parameters") or {})}
        rai_params = {**default_rai_params, **(data.get("rai_parameters") or {})}

        current_app.logger.info(">> Parameters after merging with defaults")
        current_app.logger.info(f"Environment Parameters: {environ_params}")
        current_app.logger.info(f"Model Parameters: {model_params}")
        current_app.logger.info(f"Map Parameters: {map_params}")
        current_app.logger.info(f"RAI Parameters: {rai_params}")

        # Make environment map
        try:
            map_path = make_environment_map(environ_params["world_size"], map_params)
            current_app.logger.info("make_environment_map completed")
        except Exception as e:
            current_app.logger.info(f"Error calling make_environment_map: {e}")
            raise

        # Create run_id
        run_id = round(1000 * datetime.datetime.now().timestamp(), 0)
        model_status["run_id"] = run_id
        current_app.logger.info(f"Run ID: {model_status['run_id']}")

        # Run model asynchronously
        set_model_status("running")
        current_app.logger.info(f"Map path: {map_path}")
        start_model_thread(
            map_path,
            json.dumps(environ_params),
            json.dumps(model_params),
            json.dumps(rai_params),
            run_id=model_status["run_id"]
        )

        return jsonify({'model': 'Successful run with defaults'}), 200

    except Exception as e:
        current_app.logger.error(f"Error in post_standard_run_xrai_with_defaults: {str(e)}")
        return jsonify({'model': str(e)}), 500


def run_model(temp_file_path, environ_params, model_params, rai_params, run_id):
    from app import app  # Import the Flask app instance
    from sqlalchemy.orm import scoped_session, sessionmaker
    from sqlalchemy import create_engine

    thread_logger.info("Run model process started")

    try:
        with app.app_context():  # Push the Flask application context
            # Create a local session for the thread
            engine = db.engine  # Use the same engine as Flask-SQLAlchemy
            Session = scoped_session(sessionmaker(bind=engine))
            local_session = Session()

            # Set up paths and environment
            curr_sys = platform.system()
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            venv_activate_path = os.path.join(
                project_root, '.venv_model', 'bin', 'python' if curr_sys == "Linux" else 'Scripts/python.exe'
            )
            model_script_path = os.path.join(project_root, 'models', 'xrai_runfile.py')
            env = os.environ.copy()
            env["TF_CPP_MIN_LOG_LEVEL"] = "3"
            thread_logger.info("Run variables set")

            # Construct command
            run_model_cmd = [
                venv_activate_path,
                model_script_path,
                environ_params, model_params, temp_file_path, rai_params
            ]

            # Run subprocess
            thread_logger.info("Commence model run")
            with subprocess.Popen(run_model_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, text=True,
                                  env=env) as proc:
                episode_counter = 0

                for line in proc.stdout:

                    if stop_flag.is_set():
                        set_model_status("stop")
                        thread_logger.info("Stop signal received. Terminating model run.")
                        proc.terminate()
                        break

                    # Parse episode number
                    if "episode:" in line:
                        episode_number = int(line.split("episode:")[-1].strip())
                        model_status["episode"] = episode_number
                        episode_counter += 1
                        episode_mod = episode_counter % 10
                        if episode_mod == 1:
                            success, message = batch_insertion(run_id, local_session)
                            if not success:
                                thread_logger.error(f"Batch commit failed: {message}")
                            else:
                                thread_logger.error(f"Batch committed at episode {episode_counter}")


                    thread_logger.info(f"Subprocess Output: {line.strip()}")

                for line in proc.stderr:
                    if stop_flag.is_set():
                        thread_logger.info("Stop signal received. Terminating model run.")
                        success, message = batch_insertion(run_id, local_session)
                        if not success:
                            thread_logger.error(f"Batch commit failed during stderr handling: {message}")
                        proc.terminate()
                        break
                    thread_logger.info(f"Subprocess Error: {line.strip()}")

            proc.wait()

            if stop_flag.is_set():
                model_status["status"] = "stop"
            else:
                model_status["status"] = "completed" if proc.returncode == 0 else "error"

            if os.path.exists(model_output_path):
                thread_logger.info("Performing final batch insertion.")
                success, message = batch_insertion(run_id, local_session)
                if not success:
                    thread_logger.error(f"Final batch commit failed: {message}")

            # Final updates
            closing_data = dict(
                cflt_run_id=run_id,
                cdtm_run_date=datetime.datetime.now(),
                cint_episode=episode_counter,
                cstr_model_status=model_status["status"]
            )
            update_single_table(tbl_model_runs, closing_data)

            return proc.returncode == 0

    except Exception as e:
        thread_logger.error(f"An error occurred: {str(e)}")
        if stop_flag.is_set():
            stop_flag.clear()
        return False


def start_model_thread(temp_file_path, environ_params, model_params, rai_params, run_id):
    global model_thread, stop_flag
    if model_thread is not None and model_thread.is_alive():
        return jsonify({"error": "Model is already running"}), 400
    stop_flag.clear()
    current_app.logger.info(">> Creating model thread")
    model_thread = threading.Thread(
        target=run_model,
        args=(temp_file_path, environ_params, model_params, rai_params, run_id)
    )
    current_app.logger.info(">> >> Model thread starting...")
    model_thread.start()
    return jsonify({"status": "Model started"}), 200

@main.route('/model/pause', methods=['GET'])
def pause_model():
    with open(status_path, "w") as f:
        f.write("pause")
    return jsonify({"status": "Model paused"}), 200

@main.route('/model/play', methods=['GET'])
def post_play_model():
    with open(status_path, "w") as f:
        f.write("running")
    return jsonify({"status": "Model resumed"}), 200

@main.route('/model/stop', methods=['GET'])
def stop_model():
    global stop_flag
    try:
        current_app.logger.info("Stop endpoint called.")

        # Pre-Stop Actions
        current_app.logger.info("Setting model status to 'stop'.")
        set_model_status("stop")
        model_status["status"] = "stop"
        with open(episode_path, "r") as f:
            episode = f.read().strip()
        closing_data = dict(cflt_run_id=model_status["run_id"],
                            cdtm_run_date=datetime.datetime.strptime(datetime.datetime.now().strftime("%Y%m%d_%H%M_%S"), "%Y%m%d_%H%M_%S"),
                            cint_episode=episode,
                            cstr_model_status="stopped")
        update_single_table(tbl_model_runs, closing_data)


        # Final Stop
        current_app.logger.info("Setting stop flag.")
        stop_flag.set()

        current_app.logger.info("Setting model status to 'idle'.")
        set_model_status("idle")
        model_status["status"] = "idle"

        current_app.logger.info("Model successfully stopped.")
        return jsonify({"status": "Model stopped"}), 200
    except Exception as e:
        current_app.logger.error(f"Error in stop_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

def cleanup_after_stop():
    if os.path.exists(model_output_path):
        os.remove(model_output_path)
    setup_flag_files()  # Reset flag files
    model_status.update({"status": "idle", "episode": 0, "run_id": 0})


@main.route('/model/current_episode', methods=['GET'])
def get_current_episode():
    with open(episode_path, "r") as f:
        episode = f.read().strip()
    return jsonify({'step': episode})


@main.route('/database/commit', methods=['GET'])
def db_commit():
    backup_path = "/CapstoneTeamAPI/utils/pickles/uncommitted_output.pkl"

    # Check the current database status
    with open(db_status_path, "r") as f:
        status = f.read().strip()

    if status != "commit":
        return jsonify({'db': "nothing to commit"}), 204

    try:
        # Check for uncommitted data
        if os.path.exists(backup_path):
            with open(backup_path, "rb") as f:
                uncommitted_data = pickle.load(f)
                current_app.logger.info("Found uncommitted data. Attempting to commit.")

            # Try committing the uncommitted data
            for key, value in uncommitted_data.items():
                if value is not None:
                    scalarized = pd.DataFrame(scalarize(value))
                    db.session.bulk_save_objects(
                        [tbl_utilities[key](**row) for row in scalarized.to_dict(orient='records')])
            db.session.commit()
            os.remove(backup_path)
            current_app.logger.info("Uncommitted data successfully committed.")

        # Commit current model output
        success, message = batch_insertion(model_status["run_id"])
        if not success:
            raise Exception(message)

        return jsonify({'db': "committed"}), 200
    except Exception as e:
        # Save uncommitted data to a backup file
        current_app.logger.error(f"Error during commit: {e}")
        with open(model_output_path, "rb") as f:
            model_output = pickle.load(f)
        with open(backup_path, "wb") as f:
            pickle.dump(model_output, f)
        current_app.logger.info(f"Uncommitted data saved to backup at {backup_path}")

        db.session.rollback()
        return jsonify({'error': f"Commit failed. Data saved for retry: {str(e)}"}), 500


@main.route('/model/status', methods=['GET'])
def get_current_status():
    with open(status_path, "r") as f:
        episode = f.read().strip()
    return jsonify({'status': episode})


@main.route('/database/last_run/tbl_model_runs', methods=['GET'])
def get_last_run_model_run():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        current_app.logger.info(f"Retrieved last_run_id: {last_run_id}")

        if last_run_id:
            last_model_run = db.session.query(tbl_model_runs).filter(tbl_model_runs.cflt_run_id == last_run_id).all()
            current_app.logger.info(f"Retrieved local states count: {len(last_model_run)}")

            # Serialize each row in local_states
            serialized_run = [run.__dict__ for run in last_model_run]
            # Remove the SQLAlchemy metadata key `_sa_instance_state`
            for state in serialized_run:
                state.pop('_sa_instance_state', None)

            if serialized_run:
                return jsonify(serialized_run), 200
            else:
                return jsonify({'message': 'No model run data found for the last run.'}), 204
        else:
            return jsonify({'message': 'No model run data found for the last run.'}), 204
    except Exception as e:
        current_app.logger.error(f"Error retrieving last run local state: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/database/last_run/tbl_local_state', methods=['GET'])
def get_last_run_local_state():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        current_app.logger.info(f"Retrieved last_run_id: {last_run_id}")

        if last_run_id:
            local_states = db.session.query(tbl_local_state).filter(tbl_local_state.cflt_run_id == last_run_id).all()
            current_app.logger.info(f"Retrieved local states count: {len(local_states)}")

            # Serialize each row in local_states
            serialized_states = [state.__dict__ for state in local_states]
            # Remove the SQLAlchemy metadata key `_sa_instance_state`
            for state in serialized_states:
                state.pop('_sa_instance_state', None)

            if serialized_states:
                return jsonify(serialized_states), 200
            else:
                return jsonify({'message': 'No local state data found for the last run.'}), 204
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        current_app.logger.error(f"Error retrieving last run local state: {str(e)}")
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_rewards', methods=['GET'])
def get_last_run_rewards():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        if last_run_id:
            rewards = db.session.query(tbl_rewards).filter(tbl_rewards.cflt_run_id == last_run_id).all()
            serialized_rewards = [reward.__dict__ for reward in rewards]
            for reward in serialized_rewards:
                reward.pop('_sa_instance_state', None)
            return jsonify(serialized_rewards), 200 if serialized_rewards else 204
        else:
            return jsonify({'message': 'No rewards data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/database/last_run/tbl_rai', methods=['GET'])
def get_last_run_rai():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        if last_run_id:
            all_rai = db.session.query(tbl_rai).filter(tbl_rai.cflt_run_id == last_run_id).all()
            serialized_rai_params = [rai.__dict__ for rai in all_rai]
            for rai in serialized_rai_params:
                rai.pop('_sa_instance_state', None)
            return jsonify(serialized_rai_params), 200 if serialized_rai_params else 204
        else:
            return jsonify({'message': 'No RAI data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_global_state', methods=['GET'])
def get_last_run_global_state():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        if last_run_id:
            global_states = db.session.query(tbl_global_state).filter(tbl_global_state.cflt_run_id == last_run_id).all()
            serialized_global_states = [state.__dict__ for state in global_states]
            for state in serialized_global_states:
                state.pop('_sa_instance_state', None)
            return jsonify(serialized_global_states), 200 if serialized_global_states else 204
        else:
            return jsonify({'message': 'No global state data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_drone_actions', methods=['GET'])
def get_last_run_drone_actions():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        if last_run_id:
            drone_actions = db.session.query(tbl_drone_actions).filter(tbl_drone_actions.cflt_run_id == last_run_id).all()
            serialized_drone_actions = [action.__dict__ for action in drone_actions]
            for action in serialized_drone_actions:
                action.pop('_sa_instance_state', None)
            return jsonify(serialized_drone_actions), 200 if serialized_drone_actions else 204
        else:
            return jsonify({'message': 'No drone actions data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_model_run_params', methods=['GET'])
def get_last_run_model_run_params():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        if last_run_id:
            model_run_params = db.session.query(tbl_model_run_params).filter(tbl_model_run_params.cflt_run_id == last_run_id).all()
            serialized_model_run_params = [param.__dict__ for param in model_run_params]
            for param in serialized_model_run_params:
                param.pop('_sa_instance_state', None)
            return jsonify(serialized_model_run_params), 200 if serialized_model_run_params else 204
        else:
            return jsonify({'message': 'No model run params found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_map_data', methods=['GET'])
def get_last_run_map_data():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        if last_run_id:
            map_data = db.session.query(tbl_map_data).filter(tbl_map_data.cflt_run_id == last_run_id).all()
            serialized_map_data = [data.__dict__ for data in map_data]
            for data in serialized_map_data:
                data.pop('_sa_instance_state', None)
            return jsonify(serialized_map_data), 200 if serialized_map_data else 204
        else:
            return jsonify({'message': 'No map data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@main.route('/api/get_run_id', methods=['GET'])
def api_get_run_id():
    current_app.logger.info("Getting run id...")
    print("getting run id...")
    start_time = time.time()
    try:
        run_id = db.session.execute(text('SELECT id FROM status')).scalar()
        end_time = time.time()
        query_duration = end_time - start_time
        current_app.logger.info("Query duration: %.2f seconds", query_duration)
        return jsonify({'id': run_id}), 200
    except Exception as e:
        end_time = time.time()
        query_duration = end_time - start_time
        current_app.logger.info("Query duration: %.2f seconds", query_duration)
        return jsonify({'id': 99}), 200

@main.route('/api/record_model_episode', methods=['POST'])
def api_record_model_episode():
    data = request.json
    current_episode = data.get("current_episode")

    if current_episode is None:
        return jsonify({"error": "current_episode is required"}), 400

    try:
        update_query = text("UPDATE status SET current_episode = :current_episode")
        db.session.execute(update_query, {"current_episode": current_episode})
        db.session.commit()
        current_app.logger.info(f"Episode {current_episode} recorded")
        return jsonify({"message": "Episode recorded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/database/last_run/drone_damage_count', methods=['GET'])
def get_unique_drones_with_damage():
    try:
        # Get the most recent cflt_run_id
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        current_app.logger.info(f"Retrieved last_run_id: {last_run_id}")

        if last_run_id:
            # Query unique cint_drone_id where cflt_drone_damage > 0
            unique_drones = db.session.query(tbl_local_state.cint_drone_id).distinct() \
                .filter(tbl_local_state.cflt_run_id == last_run_id) \
                .filter(tbl_local_state.cflt_drone_damage > 0).all()

            unique_drone_count = len(unique_drones)
            current_app.logger.info(f"Unique drone count with damage: {unique_drone_count}")

            return jsonify({'unique_drones_with_damage': unique_drone_count}), 200
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        current_app.logger.error(f"Error retrieving unique drones with damage: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/database/last_run/total_drone_damage', methods=['GET'])
def get_total_drone_damage():
    try:
        # Get the most recent cflt_run_id
        last_run_id = db.session.query(func.max(tbl_model_runs.cflt_run_id)).scalar()
        current_app.logger.info(f"Retrieved last_run_id: {last_run_id}")

        if last_run_id:
            # Query the sum of cflt_drone_damage
            total_damage = db.session.query(func.sum(tbl_local_state.cflt_drone_damage)) \
                .filter(tbl_local_state.cflt_run_id == last_run_id).scalar()

            total_damage = total_damage or 0  # Ensure a result of 0 if no damage rows exist
            current_app.logger.info(f"Total drone damage: {total_damage}")

            return jsonify({'total_drone_damage': total_damage}), 200
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        current_app.logger.error(f"Error retrieving total drone damage: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/database/last_n_runs/<string:table_name>', methods=['GET'])
def get_last_n_runs(table_name):
    """
    Get data for the last n distinct run IDs for a specific table.

    Args:
        table_name (str): Name of the table to query.

    Query Parameters:
        n (int): Number of recent runs to fetch (default is 1).

    Returns:
        JSON response with the requested data or an error message.
    """
    try:
        # Get the number of runs to retrieve (default to 1)
        n = request.args.get('n', default=1, type=int)

        # Map table names to actual SQLAlchemy models
        table_map = {
            "tbl_model_runs": tbl_model_runs,
            "tbl_local_state": tbl_local_state,
            "tbl_rewards": tbl_rewards,
            "tbl_global_state": tbl_global_state,
            "tbl_drone_actions": tbl_drone_actions,
            "tbl_model_run_params": tbl_model_run_params,
            "tbl_map_data": tbl_map_data,
            "tbl_rai": tbl_rai
        }

        # Validate table name
        if table_name not in table_map:
            return jsonify({"error": f"Invalid table name: {table_name}"}), 400

        # Get the table model
        table_model = table_map[table_name]

        # Query the most recent n distinct run IDs from tbl_model_runs
        run_ids = (
            db.session.query(tbl_model_runs.cflt_run_id)
            .distinct()
            .order_by(tbl_model_runs.cflt_run_id.desc())
            .limit(n)
            .all()
        )
        run_ids = [row[0] for row in run_ids]  # Extract the run IDs from the result

        if not run_ids:
            return jsonify({"message": f"No runs found for {table_name}."}), 204

        # Query data from the specified table for the retrieved run IDs
        results = (
            db.session.query(table_model)
            .filter(table_model.cflt_run_id.in_(run_ids))
            .all()
        )

        # Serialize results
        serialized_results = [row.__dict__ for row in results]
        for row in serialized_results:
            row.pop('_sa_instance_state', None)

        if not serialized_results:
            return jsonify({"message": f"No data found in {table_name} for the last {n} runs."}), 204

        return jsonify(serialized_results), 200

    except Exception as e:
        current_app.logger.error(f"Error retrieving data for {table_name}: {str(e)}")
        return jsonify({"error": str(e)}), 500
