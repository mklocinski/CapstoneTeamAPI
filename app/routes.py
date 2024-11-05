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
import datetime
import platform
import traceback
import tempfile
import time

main = Blueprint('main', __name__)
warnings.filterwarnings("ignore")
#logging.basicConfig(level=logging.INFO)



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
    current_app.logger.info("Starting standard model run:")
    data = request.get_json()
    environ_params = json.dumps(data.get("environment_parameters"))
    model_params = json.dumps(data.get("model_parameters"))
    map_params = json.dumps(data.get("map_parameters"))
    rai_params = json.dumps(data.get("rai_parameters"))
    current_app.logger.info(">> Parameters received")


    import models.MapPackage as m
    # Generate map data and write to a temporary file
    amap = m.EnvironmentMap()
    amap.generate_obstacle_data()
    map_data = amap.dataframe.reset_index(drop=True).to_json()
    current_app.logger.info(">> Map data generated")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_file:
        temp_file.write(map_data)
        temp_file_path = temp_file.name
        current_app.logger.info(f">> Map data written to temporary file: {temp_file_path}")

    # Update status table
    current_app.logger.info(">> Updating status table")
    run_id = round(1000 * datetime.datetime.now().timestamp(), 0)
    db.session.execute(text('UPDATE status SET run_id = :run_id'), {'run_id': run_id})
    db.session.execute(text("UPDATE status SET state = 'running'"))
    db.session.commit()

    # Run model and update database
    success, message = run_model(temp_file_path, environ_params, model_params, rai_params)

    # Update status table based on model run result
    db.session.execute(text('UPDATE status SET run_id = 0'))
    db.session.execute(text("UPDATE status SET state = 'complete'"))
    db.session.execute(text('UPDATE status SET episode = 0'))
    db.session.execute(text('UPDATE status SET timesteps = 0'))
    db.session.execute(text('UPDATE status SET iters = 0'))
    db.session.commit()

    # Return the appropriate response based on the outcome
    if success:
        return jsonify({'message': message}), 200
    else:
        return jsonify({'error': message}), 500


def run_model(temp_file_path, environ_params, model_params, rai_params):
    try:
        current_app.logger.info(">> Run Model")
        curr_sys = platform.system()
        current_app.logger.info(f"Current System: {curr_sys}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        current_app.logger.info(f">> >> Project Root: {project_root}")
        if curr_sys == "Linux":
            venv_activate_path = os.path.join(project_root, '.venv_model', 'bin', 'python')
        else:
            venv_activate_path = os.path.join(project_root, '.venv_model', 'Scripts', 'python.exe')

        exists = os.path.exists(venv_activate_path)
        current_app.logger.info(f">> >> .venv: {venv_activate_path}")
        current_app.logger.info(f">> >> Does {venv_activate_path} exist? {exists}")

        current_app.logger.info(environ_params)
        model_script_path = os.path.join(project_root, 'models', 'xrai_runfile.py')
        current_app.logger.info(f">> >> Model Script Path: {model_script_path}")

        run_model_cmd = [
            venv_activate_path,
            model_script_path,
            environ_params, model_params, temp_file_path, rai_params
        ]
        current_app.logger.info("Model Run Command: %s", run_model_cmd)

        with subprocess.Popen(run_model_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                current_app.logger.info("Subprocess Output: %s", line.strip())
            for line in proc.stderr:
                current_app.logger.error("Subprocess Error: %s", line.strip())

        # Check for errors in subprocess
        proc.wait()
        result_code = proc.returncode
        if result_code != 0:
            return False, "Error"

        # Remove temporary file after subprocess completes
        os.remove(temp_file_path)
        current_app.logger.info(">> Model ran successfully.")

        #current_app.logger.info([column.name for column in tbl_utilities['tbl_map_data'].__table__.columns])
        # Access the model output
        model_output_path = os.path.join(project_root, 'utils', 'pickles', 'model_output.pkl')
        if not os.path.exists(model_output_path):
            error_message = "Model output file not found."
            current_app.logger.error(error_message)
            return False, error_message

        # Load model output and commit to the database
        with open(model_output_path, 'rb') as f:
            model_output = pickle.load(f)

        for key, val in model_output.items():
            scalarized = pd.DataFrame(scalarize(val))
            current_app.logger.info(">> Loading %s to database", key)
            current_app.logger.info(scalarized.head(1))
            for ix, row in scalarized.iterrows():
                one_row = {col: row[col] for col in scalarized.columns}
                if key=="tbl_map_data":
                    with open("/CapstoneTeamAPI/utils/debug_map_data.json", "a") as debug_file:
                        debug_file.write(json.dumps(one_row, indent=4))
                        debug_file.write("\n\n")
                if key in tbl_utilities:
                    tbl_row = tbl_utilities[key](**one_row)
                    db.session.add(tbl_row)


        # Commit data to database
        db.session.commit()
        current_app.logger.info("Model output committed to the database successfully.")
        return True, "Model ran and data committed successfully."

    except Exception as e:
        # Roll back and log any errors
        db.session.rollback()
        error_message = f"An error occurred during model execution: {e}"
        traceback_info = traceback.format_exc()
        current_app.logger.error(f"{error_message}\nTraceback:\n{traceback_info}")
        return False, f"{error_message}\nTraceback:\n{traceback_info}"



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
        venv_python = "/app/.venv_model/bin/python"

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

