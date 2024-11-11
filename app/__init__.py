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

    # Apply the configuration based on the environment
    config = get_config()
    app.config.from_object(config)

    # Debugging print to verify configuration
    print("Environment:", app.config.get("FLASK_ENV"))
    print("SQLAlchemy URI in app config:", app.config.get("SQLALCHEMY_DATABASE_URI"))

    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)  # Use migrate if migrations are needed

    # Register Blueprints
    from .routes import main  # Import the main blueprint
    app.register_blueprint(main)

    # Custom CLI command to initialize the database
    @app.cli.command("init-db")
    def init_db_command():
        """Initialize the database."""
        with app.app_context():
            from .routes import init_db
            init_db()  # Call the function to initialize the database
            print("Database initialized.")


    # Configure logging
    if app.logger.hasHandlers():
        app.logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    app.logger.info("App started successfully with logging configured")

    return app




app = create_app()
