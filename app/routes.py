from flask import Blueprint, jsonify, request, current_app
from utils.database_utils import tbl_utilities, tbl_batches, scalarize
import subprocess
import os
import pickle
import pandas as pd
from sqlalchemy import text, func
import json
import warnings
from .data_models import tbl_model_runs, tbl_local_state, tbl_rewards, tbl_global_state, tbl_drone_actions, tbl_model_run_params, tbl_map_data
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
logging.basicConfig(level=logging.INFO)
thread_logger = logging.getLogger("model_thread")
completion_queue = queue.Queue()

# Flag Variables
db_status_path = "/CapstoneTeamAPI/utils/status/data_commit.txt"
episode_path = "/CapstoneTeamAPI/utils/status/model_episode.txt"
status_path = "/CapstoneTeamAPI/utils/status/model_status.txt"
model_output_path = "/CapstoneTeamAPI/utils/pickles/model_output.pkl"

@main.route('/')
def home():
    return jsonify({"message": "Welcome to the XRAI API. Please refer to the documentation for available endpoints."}), 200


def line_insertion(run_id, model_output_path):
    with open(model_output_path, 'rb') as f:
        model_output = pickle.load(f)
    for name, tbl in model_output.items():
        tbl.insert(0, "cflt_run_id", run_id)
    if os.path.exists(model_output_path):
        try:
            for key, val in model_output.items():
                scalarized = pd.DataFrame(scalarize(val))
                current_app.logger.info(">> Loading %s to database", key)
                current_app.logger.info(scalarized.columns)
                for ix, row in scalarized.iterrows():
                    one_row = {col: row[col] for col in scalarized.columns}
                    if key in tbl_utilities:
                        tbl_row = tbl_utilities[key](**one_row)
                        db.session.add(tbl_row)

            # Commit data to database
            db.session.commit()
            current_app.logger.info("Model output committed to the database successfully.")
            return True, "Model ran and data committed successfully."
        except Exception as e:
            db.session.rollback()
            error_message = f"An error occurred during database commit: {e}"
            traceback_info = traceback.format_exc()
            current_app.logger.error(f"{error_message}\nTraceback:\n{traceback_info}")
            return False, f"{error_message}\nTraceback:\n{traceback_info}"
    else:
        error_message = "Model output file not found."
        current_app.logger.error(error_message)
        return False, error_message



def make_environment_map(map_size,no_fly_zones,humans,buildings,trees,animals,
                         no_fly_zones_damage,humans_damage,buildings_damage,trees_damage,animals_damage):
    current_app.logger.info(">> Generating map data")
    import models.MapPackage as m
    amap = m.EnvironmentMap(map_size=[int(map_size),int(map_size)],
                            no_fly_zones = dict(count=int(no_fly_zones), random=True, positions=[], sizes=[], risk=no_fly_zones_damage),
                            humans = dict(count=int(humans), random=True, positions=[], sizes=[], risk=humans_damage),
                            buildings = dict(count=int(buildings), random=True, positions=[], sizes=[], risk=buildings_damage),
                            trees = dict(count=int(trees), random=True, positions=[], sizes=[], risk=trees_damage),
                            animals = dict(count=int(animals), random=True, positions=[], sizes=[], risk=animals_damage))
    amap.generate_obstacle_data()
    map_data = amap.dataframe.reset_index(drop=True).to_json()
    current_app.logger.info(">> Map data generated")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
        temp_file.write(map_data)
        temp_file_path = temp_file.name
        current_app.logger.info(f">> Map data written to temporary file: {temp_file_path}")

    return temp_file_path

def setup_flag_files():
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


def set_model_status(status):
    if not os.path.exists(status_path):
        with open(status_path, 'w') as file:
            file.write(status)
        print(f"{status_path} flag set to '{status}'.")
    else:
        with open(status_path, 'w') as file:
            file.write(status)
        print(f"{status_path} flag set to '{status}'.")




def batch_insertion(run_id):
    current_app.logger.info('>> Batch commit to database')
    with open(model_output_path, 'rb') as f:
        model_output = pickle.load(f)
    for name, tbl in model_output.items():
        current_app.logger.info(f">> Inserting run_it into {name}")
        if tbl is not None:
            tbl.insert(0, "cflt_run_id", run_id)
    if os.path.exists(model_output_path):
        try:
            for key, val in model_output.items():
                if val is not None:
                    scalarized = pd.DataFrame(scalarize(val))
                    current_app.logger.info(f">> Loading {key} to database")
                    current_app.logger.info(scalarized.columns)
                    batch = []
                    for ix, row in scalarized.iterrows():
                        one_row = {col: row[col] for col in scalarized.columns}
                        if key in tbl_utilities:
                            tbl_row = tbl_utilities[key](**one_row)
                            batch.append(tbl_row)
                    db.session.bulk_save_objects(batch)
                    db.session.commit()
                    current_app.logger.info(f">> Batch committed to {key}")

            current_app.logger.info("Model output committed to the database successfully.")
            with open(db_status_path, 'w') as file:
                file.write("queuing")
            return True, "Model ran and data committed successfully."
        except Exception as e:
            db.session.rollback()
            error_message = f"An error occurred during database commit: {e}"
            traceback_info = traceback.format_exc()
            current_app.logger.error(f"{error_message}\nTraceback:\n{traceback_info}")
            return False, f"{error_message}\nTraceback:\n{traceback_info}"
    else:
        error_message = "Model output file not found."
        current_app.logger.error(error_message)
        return False, error_message



@main.route('/model/standard/run_xrai', methods=['POST'])
def post_standard_run_xrai_system():
    current_app.logger.info("Starting standard model run:")
    set_model_status("initializing")
    try:
        current_app.logger.info(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        global model_status
        model_status["status"] = "running"

        # Set flag files
        setup_flag_files()

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
            map_path = make_environment_map(world_size.get("world_size"),
                            map_params["number_of_no-fly_zones"],
                            map_params["number_of_buildings"],
                            map_params["number_of_humans"],
                            map_params["number_of_trees"],
                            map_params["number_of_animals"],
                            map_params["no-fly_zone_collision_damages"],
                            map_params["building_collision_damages"],
                            map_params["human_collision_damage"],
                            map_params["tree_collision_damages"],
                            map_params["animal_collision_damage"]
                                            )
            current_app.logger.info("make_environment_map completed")
        except Exception as e:
            current_app.logger.info(f"Error calling make_environment_map: {e}")


        # Create run_id
        run_id = round(1000 * datetime.datetime.now().timestamp(), 0)
        model_status["run_id"] = run_id
        current_app.logger.info(model_status["run_id"])

        # Run model asynchronously
        set_model_status("running")
        start_model_thread(map_path, environ_params, model_params, rai_params, run_id=model_status["run_id"])
        return jsonify({'model': 'Successful run'}), 200
    except Exception as e:
        return jsonify({'model': str(e)}), 500



def run_model(temp_file_path, environ_params, model_params, rai_params, run_id):
    thread_logger.info("Run model process started")

    try:
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
            for line in proc.stdout:
                if "Episode:" in line:
                    episode_number = int(line.split("Episode:")[-1].strip())
                    model_status["episode"] = episode_number
                thread_logger.info(f"Subprocess Output: {line.strip()}")
            for line in proc.stderr:
                thread_logger.info(f"Subprocess Error: {line.strip()}")

        proc.wait()
        model_status["status"] = "completed" if proc.returncode == 0 else "error"
        return proc.returncode == 0
    except Exception as e:
        model_status["status"] = "error"
        thread_logger.info(f"An error occurred: {str(e)}")
        completion_queue.put({"status": "error", "error": str(e), "run_id": run_id})
        return False

def start_model_thread(temp_file_path, environ_params, model_params, rai_params, run_id):
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

@main.route('/model/current_episode', methods=['GET'])
def get_current_episode():
    with open(episode_path, "r") as f:
        episode = f.read().strip()
    return jsonify({'step': episode})

@main.route('/database/commit', methods=['GET'])
def db_commit():
    with open(db_status_path, "r") as f:
        status = f.read().strip()
        if status == "commit":
            batch_insertion(model_status["run_id"])
            return jsonify({'db': "committed"}), 200
        else:
            return jsonify({'db': "nothing to commit"}), 204

@main.route('/model/status', methods=['GET'])
def get_current_status():
    with open(status_path, "r") as f:
        episode = f.read().strip()
    return jsonify({'status': episode})

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

