from flask import Blueprint, request, jsonify
from .data_models import *
import subprocess
import os

main = Blueprint('main', __name__)

# tests, remove
@main.route('/')
def home():
    return "Hello from the home page!"

@main.route('/your_endpoint')
def your_endpoint():
    return "This is your endpoint!"

@main.route('/run_model')
def run_model_route():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    #venv_python = os.path.join(os.path.dirname(__file__), 'models', 'drlss', '.venv', 'Scripts', 'python.exe')
    venv_python = "C:/Users/mkloc/PycharmProjects/CapstoneTeamAPI/models/drlss/.venv/Scripts/python.exe"
    run_model = [
        venv_python,
        '-c',
        'from models.WrappedModel import main; main()'
    ]

    # Run the subprocess and capture output
    result = subprocess.run(run_model, capture_output=True, text=True)

    # Check if the model returned an error
    if result.returncode != 0:
        return jsonify({'error': result.stderr}), 500

    try:
        model_output = json.loads(result.stdout)  # Adjust this if output is not JSON
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Failed to parse model output', 'details': str(e)}), 500

    # Process the model output as before
    tb1 = tbl_model_runs(**model_output['tbl_model_runs'])
    tb2 = tbl_model_run_params(**model_output['tbl_model_run_params'])
    tb3 = tbl_local_state(**model_output['tbl_local_state'])
    tb4 = tbl_global_state(**model_output['tbl_global_state'])
    tb5 = tbl_rewards(**model_output['tbl_rewards'])
    tb6 = tbl_drone_actions(**model_output['tbl_drone_actions'])

    # Add to the database
    for tbl in [tb1, tb2, tb3, tb4, tb5, tb6]:
        db.session.add(tbl)

    db.session.commit()

    return jsonify({'result': model_output}), 201


@main.route('/last_model_run', methods=['GET'])
def last_model_run():
    outputs = dm.tbl_model_runs.query.all()
    return jsonify([{'input': output.user_input, 'result': output.model_result} for output in outputs]), 200