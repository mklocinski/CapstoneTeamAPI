import os
import platform
from urllib.parse import urlparse
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
from .data_models import db  # Import db from data_models directly
from .config import get_config  # Import the config loader
import logging
import warnings
import threading

# Load environment variables from .env or .env.local
load_dotenv()
if platform.system() == "Linux":
    load_dotenv(".env")
else:
    load_dotenv(".env.local")

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=Warning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")



def create_app():
    app = Flask(__name__)
    print("check 1")
    # Apply the configuration based on the environment
    config = get_config()
    app.config.from_object(config)
    print("check 2")
    # Debugging print to verify configuration
    print("Environment:", app.config.get("FLASK_ENV"))
    print("SQLAlchemy URI in app config:", app.config.get("SQLALCHEMY_DATABASE_URI"))
    print("check 3")
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)  # Use migrate if migrations are needed
    print("check 4")
    # Register Blueprints
    from .routes import main  # Import the main blueprint
    app.register_blueprint(main)
    print("check 5")
    # # Custom CLI command to initialize the database
    # @app.cli.command("init-db")
    # def init_db_command():
    #     """Initialize the database."""
    #     with app.app_context():
    #         from .routes import init_db
    #         init_db()  # Call the function to initialize the database
    #         print("Database initialized.")

    print("check 6")
    # Configure logging
    if app.logger.hasHandlers():
        app.logger.handlers.clear()
    print("check 7")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    print("check 8")
    # Flag Variables
    db_status_path = "/CapstoneTeamAPI/utils/status/data_commit.txt"
    episode_path = "/CapstoneTeamAPI/utils/status/model_episode.txt"
    status_path = "/CapstoneTeamAPI/utils/status/model_status.txt"
    print("check 9")
    # if os.path.exists(db_status_path):
    #     with open(db_status_path, "w") as f:
    #         f.write("queuing")
    # print("check 10")
    # if os.path.exists(episode_path):
    #     with open(episode_path, "w") as f:
    #         f.write("0")
    # print("check 11")
    # if os.path.exists(status_path):
    #     with open(status_path, "w") as f:
    #         f.write("idle")
    # print("check 12")

    app.logger.info("App started successfully with logging configured")

    return app




app = create_app()
