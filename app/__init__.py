import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from dotenv import load_dotenv
from .data_models import db  # Now importing db from data_models directly
from .config import get_config  # Import the config loader

load_dotenv()  # Load environment variables from .env
print("Environment variable check: SQLALCHEMY_DATABASE_URI =", os.getenv("SQLALCHEMY_DATABASE_URI"))

def create_app():
    app = Flask(__name__)

    # Apply the corresponding configuration based on the environment
    config = get_config()
    app.config.from_object(config)

    # Debugging print to verify configuration
    print("SQLAlchemy URI in app config:", app.config.get("SQLALCHEMY_DATABASE_URI"))

    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)

    # Register Blueprints
    from .routes import main
    app.register_blueprint(main)

    @app.cli.command("init-db")
    def init_db_command():
        """Initialize the database."""
        with app.app_context():
            from .routes import init_db
            init_db()
            print("Database initialized.")

    return app

# Initialize the app
app = create_app()
