from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    db = SQLAlchemy()
    migrate = Migrate()

    # Initialize SQLAlchemy and Flask-Migrate
    db.init_app(app)
    migrate.init_app(app, db)

    # Import your models so they are registered with SQLAlchemy
    from . import data_models

    # Import routes from another module (routes.py)
    from .routes import main
    app.register_blueprint(main)

    return app
