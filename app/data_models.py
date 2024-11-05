from flask import Flask
from flask_migrate import Migrate
import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class tbl_status(db.Model):
    __tablename__ = 'status'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Float, primary_key=False)
    state = db.Column(db.String(255), nullable=False)  # e.g., 'running', 'paused'
    episode = db.Column(db.Integer, nullable=False, default=0)  # Track episode number
    timesteps =db.Column(db.Integer, nullable=False, default=0)
    iters = db.Column(db.Integer, nullable=False, default=0)


    def __repr__(self):
        return f"<Status(state='{self.state}', episode={self.episode})>"


class tbl_model_runs(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cdtm_run_date = db.Column(db.Date, unique=False, nullable=False)
    cbln_terminal_episode = db.Column(db.Boolean, unique=False, nullable=False)


class tbl_model_run_params(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cstr_environment_id = db.Column(db.String, unique=False, nullable=False)
    cint_timesteps_per_batch = db.Column(db.Integer, unique=False, nullable=False)
    cflt_max_kl = db.Column(db.Float, unique=False, nullable=False)
    cint_cg_iters = db.Column(db.Integer, unique=False, nullable=False)
    cflt_cg_damping = db.Column(db.Float, unique=False, nullable=False)
    # cint_max_timesteps = db.Column(db.Integer, unique=False, nullable=True)
    cflt_gamma = db.Column(db.Float, unique=False, nullable=False)
    cflt_lam = db.Column(db.Float, unique=False, nullable=False)
    cint_vf_iters = db.Column(db.Integer, unique=False, nullable=False)
    cint_vf_stepsize = db.Column(db.Integer, unique=False, nullable=False)
    cint_nr_agents = db.Column(db.Integer, unique=False, nullable=False)
    cstr_obs_mode = db.Column(db.String, unique=False, nullable=False)
    cint_comm_radius = db.Column(db.Integer, unique=False, nullable=False)
    cint_world_size = db.Column(db.Integer, unique=False, nullable=False)
    cint_distance_bins = db.Column(db.Integer, unique=False, nullable=False)
    cint_bearing_bins = db.Column(db.Integer, unique=False, nullable=False)
    cbln_torus = db.Column(db.Boolean, unique=False, nullable=False)
    cstr_dynamics = db.Column(db.String, unique=False, nullable=False)

class tbl_local_state(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cint_episode_id = db.Column(db.Integer, primary_key=False)
    cint_drone_id = db.Column(db.Integer, primary_key=False)
    cflt_x_coord = db.Column(db.Float, unique=False, nullable=False)
    cflt_y_coord = db.Column(db.Float, unique=False, nullable=False)
    cflt_orientation = db.Column(db.Float, unique=False, nullable=False)
    cflt_linear_velocity = db.Column(db.Float, unique=False, nullable=False)
    cflt_angular_velocity = db.Column(db.Float, unique=False, nullable=False)
    cint_drone_collisions = db.Column(db.Integer, primary_key=False)
    cflt_drone_obstacle_distance = db.Column(db.Integer, primary_key=False)

class tbl_global_state(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cint_episode_id = db.Column(db.Integer, primary_key=False)
    cstr_state_encoding = db.Column(db.String, unique=False, nullable=False)

class tbl_drone_actions(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cint_episode_id = db.Column(db.Integer, primary_key=False)
    cint_drone_id = db.Column(db.Integer, primary_key=False)
    cflt_linear_velocity = db.Column(db.Float, unique=False, nullable=False)
    cflt_angular_velocity = db.Column(db.Float, unique=False, nullable=False)


class tbl_rewards(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cint_episode_id = db.Column(db.Integer, primary_key=False)
    cflt_reward = db.Column(db.Float, unique=False, nullable=False)
    cflt_distance_reward = db.Column(db.Float, unique=False)
    cflt_action_penalty = db.Column(db.Float, unique=False)


class tbl_map_data(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cflt_x_coord = db.Column(db.Float, unique=False)
    cflt_y_coord = db.Column(db.Float, unique=False)
    cstr_point_type = db.Column(db.String, unique=False)
    cint_obstacle_id = db.Column(db.Integer, unique=False)
    cstr_obstacle = db.Column(db.String, unique=False)
    cflt_obstacle_risk = db.Column(db.Float, unique=False)
    cstr_obstacle_color = db.Column(db.String, unique=False)

class tbl_rai(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cflt_run_id = db.Column(db.Float, primary_key=False)
    cbln_basic_collision_avoidance = db.Column(db.Boolean, unique=False)
    cflt_basic_collision_avoidance_penalty = db.Column(db.Float, unique=False)
    cbln_advanced_collision_avoidance = db.Column(db.Boolean, unique=False)
    cflt_advanced_collision_buffer_distance = db.Column(db.Float, unique=False)
    cflt_advanced_collision_avoidance_penalty = db.Column(db.Float, unique=False)
    cbln_basic_damage_avoidance = db.Column(db.Boolean, unique=False)

# if __name__ == "__main__":
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)
