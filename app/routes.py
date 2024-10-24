from flask import Blueprint, current_app, jsonify, request

#from deep_rl_for_swarms.common.misc_util import pickle_load
from .data_models import *
from models.ModelOutputWrapper import scalarize
from utils.database_utils import tbl_utilities
import subprocess
import os
import pickle
import pandas as pd
from sqlalchemy import text
import json
import warnings
import sqlite3
import multiprocessing

main = Blueprint('main', __name__)
warnings.filterwarnings("ignore")


def init_db():
    conn = sqlite3.connect('model_status.db')
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS status (state TEXT, episode INTEGER)''')

    # Check if the table is empty, and initialize with "running" status
    c.execute('''SELECT COUNT(*) FROM status''')
    if c.fetchone()[0] == 0:
        c.execute('''INSERT INTO status (state, episode) VALUES ("running", 0)''')
    else:
        try:
            c.execute('''SELECT episode FROM status LIMIT 1''')
        except sqlite3.OperationalError:
            c.execute('''ALTER TABLE status ADD COLUMN episode INTEGER''')
            c.execute('''UPDATE status SET episode = 0 WHERE state = "running"''')

    conn.commit()
    conn.close()

def check_for_data_batches():
    batch_number = 0

    while True:
        file = 'utils\pickles\model_output.pkl'
        if os.path.exists(file):
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
                    for row in result:
                        print(row)
                except Exception as e:
                    db.session.rollback()  # Rollback in case of error
                    print(f"Error during commit: {e}")
                finally:
                    db.session.close()

        else:
            break


# Run this code before the first request is processed
init_db()
@main.route('/database/start_batch_processing', methods=['POST'])
def start_batch_processing():
    process = multiprocessing.Process(target=check_for_data_batches)
    process.start()
    return "Batch processing started"


@main.route('/model/standard/run_xrai', methods=['GET','POST'])
def post_model_run():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        venv_python = "C:/Users/mkloc/anaconda3/envs/drlapi_venv/python.exe"

        run_model = [
            venv_python,
            os.path.join(project_root, 'models', 'WrappedModel.py')
        ]

        result = subprocess.run(run_model, capture_output=True, text=True)
        print("RESULT! " + str(result.returncode))
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

        if result.returncode == 0:
            if os.path.exists('utils\pickles\model_output.pkl'):
                with open('utils\pickles\model_output.pkl', 'rb') as f:
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

                try:
                    db.session.commit()
                    print('Committed to database')

                    # Query the database before closing the session
                    result = db.session.query(tbl_utilities["tbl_model_runs"]).all()
                    print(f"Rows in database for tbl_model_runs: {len(result)}")
                    for row in result:
                        print(row)
                except Exception as e:
                    db.session.rollback()  # Rollback in case of error
                    print(f"Error during commit: {e}")
                finally:
                    db.session.close()

            return jsonify({'model_output': "Model ran successfully; output committed to database"})

        else:
            return jsonify({'error': result.stderr}), 500

@main.route('/model/live/run_xrai', methods=['POST'])
def run_xrai_system():

    data = request.get_json()
    environ_params = json.dumps(data.get("environment_parameters"))
    model_params = json.dumps(data.get("model_parameters"))
    rai_params = json.dumps(data.get("rai_parameters"))
    map_params = json.dumps(data.get("map_parameters"))


    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    venv_python = "C:/Users/mkloc/anaconda3/envs/drlapi_venv/python.exe"

    conn = sqlite3.connect('model_status.db')
    c = conn.cursor()
    c.execute('''SELECT state FROM status''')
    status = c.fetchone()[0]

    if status == "paused":
        c.execute('''UPDATE status SET state = "running"''')
        conn.commit()
        conn.close()
        current_state = "utils\pickles\checkpoint.pkl"
        current_count = "utils\pickles\checkpoint_counters.pkl"
        command = [
            venv_python,
            os.path.join(project_root, 'models', 'xrai_runfile.py'),
            environ_params, model_params, map_params, [current_state, current_count]
        ]
    else:
        conn.close()
        command = [
            venv_python,
            os.path.join(project_root, 'models', 'xrai_runfile.py'),
            environ_params, model_params, map_params
        ]

    try:
        print("Running Model...")
        model_status = {"status": "running"}
        with open("utils\pickles\model_status.pkl", "wb") as f:
            pickle.dump(model_status, f)
        result = subprocess.run(command, capture_output=True, text=True)
        print("RESULT! " + str(result.returncode))
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")

        if result.returncode == 0:
            if os.path.exists('utils\pickles\model_output.pkl'):
                with open('utils\pickles\model_output.pkl', 'rb') as f:
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


# @main.route('/model/status', methods=['GET'])
# def get_model_status():
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     venv_python = "C:/Users/mkloc/anaconda3/envs/drlapi_venv/python.exe"  # Adjust this to the path of your Conda installation
#
#     run_model = [
#         venv_python,
#         os.path.join(project_root, 'models', 'WrappedModel.py')
#     ]
#
#     result = subprocess.run(run_model, capture_output=True, text=True)

@main.route(rule='/model/pause', methods=['POST'])
def post_pause_model():
    conn = sqlite3.connect('model_status.db')
    c = conn.cursor()
    c.execute('''UPDATE status SET state = "paused"''')
    conn.commit()
    conn.close()
    return jsonify({"Status":"Paused"})

@main.route(rule='/model/play', methods=['POST'])
def post_play_model():
    model_status = {"status": "running"}
    with open("utils\pickles\model_status.pkl", "wb") as f:
        pickle.dump(model_status, f)
    return jsonify(model_status)


@main.route('/database/last_run/tbl_model_runs', methods=['GET'])
def get_last_run_model_run():
    query_string = text("""
    SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id
        
    """)
    result = db.session.execute(query_string)
    print(result)
    query_output = [dict(row._mapping) for row in result]

    try:
        if query_output:
            return jsonify(query_output), 200
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/database/last_run/tbl_local_state', methods=['GET'])
def get_last_run_local_state():
        query_string = text("""
        SELECT ls.*
        FROM tbl_local_state ls INNER JOIN 
            (SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id) last_run
            ON ls.cflt_run_id = last_run.cflt_run_id
        """)
        result = db.session.execute(query_string)
        print(result)
        query_output = [dict(row._mapping) for row in result]

        try:
            if query_output:
                return jsonify(query_output), 200
            else:
                return jsonify({'message': 'No local state data found for the last run.'}), 204
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@main.route('/database/last_run/tbl_rewards', methods=['GET'])
def get_last_run_rewards():
        query_string = text("""
        SELECT rs.*
        FROM tbl_rewards rs INNER JOIN 
            (SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id) last_run
            ON rs.cflt_run_id = last_run.cflt_run_id
        """)
        result = db.session.execute(query_string)
        print(result)
        query_output = [dict(row._mapping) for row in result]

        try:
            if query_output:
                return jsonify(query_output), 200
            else:
                return jsonify({'message': 'No local state data found for the last run.'}), 204
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@main.route('/database/last_run/tbl_global_state', methods=['GET'])
def get_last_run_global_state():
        query_string = text("""
        SELECT gs.*
        FROM tbl_global_state gs INNER JOIN 
            (SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id) last_run
            ON gs.cflt_run_id = last_run.cflt_run_id
        """)
        result = db.session.execute(query_string)
        print(result)
        query_output = [dict(row._mapping) for row in result]

        try:
            if query_output:
                return jsonify(query_output), 200
            else:
                return jsonify({'message': 'No local state data found for the last run.'}), 204
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_drone_actions', methods=['GET'])
def get_last_run_drone_actions():
    query_string = text("""
        SELECT ds.*
        FROM tbl_drone_actions ds INNER JOIN 
            (SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id) last_run
            ON ds.cflt_run_id = last_run.cflt_run_id
        """)
    result = db.session.execute(query_string)
    print(result)
    query_output = [dict(row._mapping) for row in result]

    try:
        if query_output:
            return jsonify(query_output), 200
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main.route('/database/last_run/tbl_model_run_params', methods=['GET'])
def get_last_run_model_run_params():
    query_string = text("""
        SELECT ps.*
        FROM tbl_model_run_params ps INNER JOIN 
            (SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id) last_run
            ON ps.cflt_run_id = last_run.cflt_run_id
        """)
    result = db.session.execute(query_string)
    print(result)
    query_output = [dict(row._mapping) for row in result]

    try:
        if query_output:
            return jsonify(query_output), 200
        else:
            return jsonify({'message': 'No local state data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@main.route('/database/last_run/tbl_map_data', methods=['GET'])
def get_last_run_map_data():
    query_string = text("""
        SELECT md.*
        FROM tbl_map_data md INNER JOIN 
            (SELECT mr.*
    FROM tbl_model_runs mr INNER JOIN 
        (SELECT MAX(r.id) AS last_id 
        FROM tbl_model_runs r) last_run
    ON mr.id = last_run.last_id) last_run
            ON md.cflt_run_id = last_run.cflt_run_id
        """)
    result = db.session.execute(query_string)
    print(result)
    query_output = [dict(row._mapping) for row in result]

    try:
        if query_output:
            return jsonify(query_output), 200
        else:
            return jsonify({'message': 'No map data found for the last run.'}), 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/model/current_episode', methods=['GET'])
def get_current_episode():
    conn = sqlite3.connect('model_status.db')
    c = conn.cursor()
    c.execute('''SELECT state FROM status''')
    current_episode = c.fetchone()[0]
    return jsonify({"episode": current_episode})