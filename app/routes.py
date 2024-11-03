from flask import Blueprint, current_app, jsonify, request, copy_current_request_context
from utils.database_utils import scalarize
from utils.database_utils import tbl_utilities
import subprocess
import os
import pickle
import pandas as pd
from sqlalchemy import text, func
import json
import warnings
import multiprocessing
from .data_models import tbl_model_runs, tbl_local_state, tbl_rewards, tbl_global_state, tbl_drone_actions, tbl_model_run_params, tbl_map_data
from . import db
import tempfile
import datetime
import platform
import sys

main = Blueprint('main', __name__)
warnings.filterwarnings("ignore")


def init_db():
    with current_app.app_context():
        # Initialize the database tables if they don't exist
        db.create_all()

        # Ensure the status table is initialized with a default state
        # Checking for existence before inserting
        status_exists = db.session.execute(
            text('SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name=\'status\')')).scalar()
        if status_exists:
            status_entry = db.session.execute(text('SELECT COUNT(*) FROM status')).scalar()
            if status_entry == 0:
                db.session.execute(text("INSERT INTO status (state, episode) VALUES ('running', 0)"))
                db.session.commit()

@main.route('/')
def home():
    return jsonify({"message": "Welcome to the XRAI API. Please refer to the documentation for available endpoints."}), 200

def check_for_data_batches():
    batch_number = 0
    file = 'utils/pickles/model_output.pkl'

    while os.path.exists(file):
        with open(file, 'rb') as f:
            batched_data = pickle.load(f)

        with current_app.app_context():
            for key, val in batched_data.items():
                print(f'Processing {key}...')
                scalarized = pd.DataFrame(scalarize(val))
                print(f"Length of scalarized data: {len(scalarized)}")

                for ix, row in scalarized.iterrows():
                    one_row = {col: row[col] for col in scalarized.columns}
                    tbl_row = tbl_utilities[key](**one_row)
                    db.session.add(tbl_row)

            try:
                db.session.commit()
                print('Committed to database')
                # Query the database before closing the session
                result = db.session.query(tbl_utilities["tbl_model_runs"]).all()
                print(f"Rows in database for tbl_model_runs: {len(result)}")
            except Exception as e:
                db.session.rollback()  # Rollback in case of error
                print(f"Error during commit: {e}")
            finally:
                db.session.close()

        break


@main.route('/database/start_batch_processing', methods=['POST'])
def start_batch_processing():
    process = multiprocessing.Process(target=check_for_data_batches)
    process.start()
    return "Batch processing started"


@main.route('/model/standard/run_xrai', methods=['POST'])
def post_standard_run_xrai_system():
    print("start run")
    data = request.get_json()
    environ_params = json.dumps(data.get("environment_parameters"))
    model_params = json.dumps(data.get("model_parameters"))
    map_params = json.dumps(data.get("map_parameters"))
    rai_params = json.dumps(data.get("rai_parameters"))

    import models.MapPackage as m
    map = m.EnvironmentMap()
    map.generate_obstacle_data()
    map_data = map.dataframe.reset_index(drop=True).to_json()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
        temp_file.write(map_data)
        temp_file_path = temp_file.name

    run_id = round(1000 * datetime.datetime.now().timestamp(), 0)
    db.session.execute(text('UPDATE status SET run_id = :run_id'), {'run_id': run_id})
    db.session.execute(text("UPDATE status SET state = 'running'"))
    db.session.commit()

    def run_model():
        with current_app.app_context():
            # Determine project root and set the virtual environment's Python executable
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            #venv_python = "/app/model_venv/bin/python3.7"
            venv_python = sys.executable
            # if platform.system() == "Windows":
            #     venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
            # else:
            #     venv_python = "/app/model_venv/bin/python3.7"
            print(platform.system())
            print(venv_python)
            model_script_path = os.path.join(project_root, 'models', 'xrai_runfile.py')

            run_model = [
                venv_python,
                model_script_path,
                environ_params, model_params, temp_file_path, rai_params
            ]

            # Run the subprocess and capture output
            result = subprocess.run(run_model, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Model execution failed: {result.stderr}")
                return None  # Handle error if needed

            os.remove(temp_file_path)

            # Access the model output
            model_output_path = os.path.join(project_root, 'utils', 'pickles', 'model_output.pkl')
            if os.path.exists(model_output_path):
                with open(model_output_path, 'rb') as f:
                    model_output = pickle.load(f)

                # Database interactions within the same context
                for key, val in model_output.items():
                    scalarized = pd.DataFrame(scalarize(val))
                    for ix, row in scalarized.iterrows():
                        one_row = {col: row[col] for col in scalarized.columns}
                        if key in tbl_utilities:
                            tbl_row = tbl_utilities[key](**one_row)
                            db.session.add(tbl_row)
                db.session.commit()
                print("Model output committed to the database successfully.")

    # Call the model-running function
    run_model()

    db.session.execute(text('UPDATE status SET run_id = 0'))
    db.session.execute(text("UPDATE status SET state = 'complete'"))
    db.session.execute(text('UPDATE status SET episode = 0'))
    db.session.execute(text('UPDATE status SET timesteps = 0'))
    db.session.execute(text('UPDATE status SET iters = 0'))
    db.session.commit()

    return jsonify({'message': 'Model run initiated; output will be stored upon completion.'}), 200

@main.route('/model/live/run_xrai', methods=['POST'])
def post_live_run_xrai_system():
    data = request.get_json()
    environ_params = json.dumps(data.get("environment_parameters"))
    model_params = json.dumps(data.get("model_parameters"))
    rai_params = json.dumps(data.get("rai_parameters"))
    map_params = json.dumps(data.get("map_parameters"))

    @copy_current_request_context
    def run_model():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        venv_python = "/app/model_venv/bin/python"

        # Check the current state of the model from the database
        status = db.session.execute(text('SELECT state FROM status')).scalar()

        if status == "paused":
            db.session.execute(text('UPDATE status SET state = "running"'))
            db.session.commit()
            current_state = "utils/pickles/checkpoint.pkl"
            current_count = "utils/pickles/checkpoint_counters.pkl"
            command = [
                venv_python,
                os.path.join(project_root, 'models', 'xrai_runfile.py'),
                environ_params, model_params, map_params, [current_state, current_count]
            ]
        else:
            command = [
                venv_python,
                os.path.join(project_root, 'models', 'xrai_runfile.py'),
                environ_params, model_params, map_params
            ]

        try:
            print("Running Model...")
            model_status = {"status": "running"}
            with open("utils/pickles/model_status.pkl", "wb") as f:
                pickle.dump(model_status, f)
            result = subprocess.run(command, capture_output=True, text=True)
            print("RESULT! " + str(result.returncode))
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")

            if result.returncode == 0:
                if os.path.exists('utils/pickles/model_output.pkl'):
                    with open('utils/pickles/model_output.pkl', 'rb') as f:
                        model_output = pickle.load(f)
                else:
                    print("model_output.pkl not found!")
                    return jsonify({'error': 'Model output file not found!'}), 500

                with current_app.app_context():
                    for key, val in model_output.items():
                        print(f'Processing {key}...')
                        scalarized = pd.DataFrame(scalarize(val))
                        print(f"Length of scalarized data: {len(scalarized)}")
                        for ix, row in scalarized.iterrows():
                            one_row = {col: row[col] for col in scalarized.columns}
                            tbl_row = tbl_utilities[key](**one_row)
                            db.session.add(tbl_row)
                    print("Finished processing data")
                    try:
                        db.session.commit()
                        print('Committed to database')
                        test_result = db.session.query(tbl_utilities["tbl_model_runs"]).all()
                        print(f"Rows in database for tbl_model_runs: {len(test_result)}")
                        for row in test_result:
                            print(row)
                    except Exception as e:
                        db.session.rollback()  # Rollback in case of error
                        print(f"Error during commit: {e}")
                    finally:
                        db.session.close()

                return jsonify({'status': 'success', 'output': result.stdout})
            else:
                return jsonify({'status': 'error', 'message': result.stderr}), 500
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    # Call the model-running function
    run_model()


@main.route('/model/pause', methods=['POST'])
def post_pause_model():
    db.session.execute(text('UPDATE status SET state = "paused"'))
    db.session.commit()
    return jsonify({"Status": "Paused"})


@main.route('/model/play', methods=['POST'])
def post_play_model():
    model_status = {"status": "running"}
    with open("utils/pickles/model_status.pkl", "wb") as f:
        pickle.dump(model_status, f)
    return jsonify(model_status)



@main.route('/model/current_episode', methods=['GET'])
def get_current_episode():
    try:
        status_entry = db.session.query(tbl_model_runs.cflt_run_id).order_by(tbl_model_runs.id.desc()).first()
        if status_entry:
            return jsonify({"episode": status_entry}), 200
        else:
            return jsonify({'message': 'No current episode data found.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@main.route('/database/last_run/tbl_local_state', methods=['GET'])
def get_last_run_local_state():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.id)).scalar()
        if last_run_id:
            local_states = db.session.query(tbl_local_state).filter(tbl_local_state.cflt_run_id == last_run_id).all()
            return jsonify([state for state in local_states]), 200 if local_states else 204
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_rewards', methods=['GET'])
def get_last_run_rewards():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.id)).scalar()
        if last_run_id:
            rewards = db.session.query(tbl_rewards).filter(tbl_rewards.cflt_run_id == last_run_id).all()
            return jsonify([reward for reward in rewards]), 200 if rewards else 204
        else:
            return jsonify({'message': 'No rewards data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_global_state', methods=['GET'])
def get_last_run_global_state():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.id)).scalar()
        if last_run_id:
            global_states = db.session.query(tbl_global_state).filter(tbl_global_state.cflt_run_id == last_run_id).all()
            return jsonify([state for state in global_states]), 200 if global_states else 204
        else:
            return jsonify({'message': 'No global state data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_drone_actions', methods=['GET'])
def get_last_run_drone_actions():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.id)).scalar()
        if last_run_id:
            drone_actions = db.session.query(tbl_drone_actions).filter(tbl_drone_actions.cflt_run_id == last_run_id).all()
            return jsonify([action for action in drone_actions]), 200 if drone_actions else 204
        else:
            return jsonify({'message': 'No drone actions data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_model_run_params', methods=['GET'])
def get_last_run_model_run_params():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.id)).scalar()
        if last_run_id:
            model_run_params = db.session.query(tbl_model_run_params).filter(tbl_model_run_params.cflt_run_id == last_run_id).all()
            return jsonify([param for param in model_run_params]), 200 if model_run_params else 204
        else:
            return jsonify({'message': 'No model run params found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_map_data', methods=['GET'])
def get_last_run_map_data():
    try:
        last_run_id = db.session.query(func.max(tbl_model_runs.id)).scalar()
        if last_run_id:
            map_data = db.session.query(tbl_map_data).filter(tbl_map_data.cflt_run_id == last_run_id).all()
            return jsonify([data for data in map_data]), 200 if map_data else 204
        else:
            return jsonify({'message': 'No map data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/api/check_model_status', methods=['GET'])
def api_check_model_status():
    try:
        status = db.session.execute(text('SELECT state FROM status')).scalar()
        return jsonify({'status': status}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        return jsonify({"message": "Episode recorded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500