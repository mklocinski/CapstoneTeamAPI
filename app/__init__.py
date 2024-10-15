from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # Import routes from another module (routes.py)
    from .routes import main

    # Register blueprint or just define routes here
    app.register_blueprint(main)

    return app
