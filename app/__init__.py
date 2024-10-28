from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .data_models import db  # Now importing db from data_models directly
from .config import DevelopmentConfig, ProductionConfig  # Import both configs
import os

def create_app():
    app = Flask(__name__)

    # Check the environment and apply the corresponding configuration
    if os.getenv("FLASK_ENV") == "development":
        app.config.from_object(DevelopmentConfig)
    else:
        app.config.from_object(ProductionConfig)

    db.init_app(app)  # Initialize db here
    migrate = Migrate(app, db)

    import pandas as pd
    import numpy as np
    print(f"Startup Check - Pandas version: {pd.__version__}")
    print(f"Startup Check - Numpy version: {np.__version__}")

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

# Initialize the app
app = create_app()
