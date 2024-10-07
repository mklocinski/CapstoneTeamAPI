from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)


class tbl_model_runs(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    run_date = db.Column(db.Date, unique=False, nullable=False)
    terminal_episode = db.Column(db.Integer, unique=False, nullable=False)

    def __repr__(self):
        return f'<Model run {self.run_id} executed on {self.run_date}; terminated at episode {self.terminal_episode}>'

class tbl_model_run_params(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    enviroment_id = db.Column(db.String, unique=False, nullable=False)
    timesteps_per_batch = db.Column(db.Integer, unique=False, nullable=False)
    max_kl = db.Column(db.Float, unique=False, nullable=False)
    cg_iters = db.Column(db.Integer, unique=False, nullable=False)
    cg_damping = db.Column(db.Float, unique=False, nullable=False)
    max_timesteps = db.Column(db.Integer, unique=False, nullable=False)
    gamma = db.Column(db.Float, unique=False, nullable=False)
    lam = db.Column(db.Float, unique=False, nullable=False)
    vf_iters = db.Column(db.Integer, unique=False, nullable=False)
    vf_stepsize = db.Column(db.Integer, unique=False, nullable=False)
    nr_agents = db.Column(db.Integer, unique=False, nullable=False)
    obs_mode = db.Column(db.String, unique=False, nullable=False)
    comm_radius = db.Column(db.Integer, unique=False, nullable=False)
    world_size = db.Column(db.Integer, unique=False, nullable=False)
    distance_bins = db.Column(db.Integer, unique=False, nullable=False)
    bearing_bins = db.Column(db.Integer, unique=False, nullable=False)
    torus = db.Column(db.Integer, unique=False, nullable=False)
    dynamics = db.Column(db.String, unique=False, nullable=False)

class tbl_local_state(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    episode_id = db.Column(db.Integer, primary_key=True)
    drone_id = db.Column(db.Integer, primary_key=True)
    x_coord = db.Column(db.Float, unique=False, nullable=False)
    y_coord = db.Column(db.Float, unique=False, nullable=False)
    orientation = db.Column(db.Float, unique=False, nullable=False)
    linear_velocity = db.Column(db.Float, unique=False, nullable=False)
    angular_velocity = db.Column(db.Float, unique=False, nullable=False)

class tbl_global_state(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    episode_id = db.Column(db.Integer, primary_key=True)
    state_encoding = db.Column(db.ARRAY(db.Integer), unique=False, nullable=False)

class tbl_drone_actions(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    episode_id = db.Column(db.Integer, primary_key=True)
    drone_id = db.Column(db.Integer, primary_key=True)
    linear_velocity = db.Column(db.Float, unique=False, nullable=False)
    angular_velocity = db.Column(db.Float, unique=False, nullable=False)


class tbl_rewards(db.Model):
    run_id = db.Column(db.Integer, primary_key=True)
    episode_id = db.Column(db.Integer, primary_key=True)
    reward = db.Column(db.Float, unique=False, nullable=False)



if __name__ == "__main__":
    app.run(debug=True)
