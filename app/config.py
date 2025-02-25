import os
import platform

class Config:
    """Base configuration with default settings shared across environments."""
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = False

class ProductionConfig(Config):
    uri = os.getenv("HEROKU_POSTGRESQL_CYAN_URL") or os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")

    # Convert to SQLAlchemy-compatible URI if necessary
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

    SQLALCHEMY_DATABASE_URI = uri
    DEBUG = False


class DevelopmentConfig(Config):
    uri = os.getenv("SQLALCHEMY_DATABASE_URI", "postgresql://myuser:mypassword@db:5432/mylocaldb")

    # Convert to SQLAlchemy-compatible URI if necessary
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

    SQLALCHEMY_DATABASE_URI = uri
    DEBUG = True  # Enable debugging for development

def get_config():
    """Environment-specific configuration loader for convenience."""
    curr_sys = platform.system()
    env = os.getenv("FLASK_ENV", "development").lower()
    print(f'FLASK_ENV in get_config(): {env}')
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
