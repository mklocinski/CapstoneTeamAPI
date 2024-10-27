from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .data_models import db  # Now importing db from data_models directly
from .config import Config
import os


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)  # Initialize db here
    migrate = Migrate(app, db)

    # Import routes and register Blueprints
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


app = create_app()