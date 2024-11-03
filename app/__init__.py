import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
from .data_models import db  # Import db from data_models directly
from .config import get_config  # Import the config loader

# Load environment variables from .env
load_dotenv()
print("Environment:",  os.getenv("FLASK_ENV"))
print("Environment variable check: SQLALCHEMY_DATABASE_URI =", os.getenv("SQLALCHEMY_DATABASE_URI"))

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

    return app

app = create_app()
