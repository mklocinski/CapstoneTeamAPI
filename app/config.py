import os


class Config:
    """Base configuration with default settings shared across environments."""
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disables modification tracking to save memory
    FLASK_ENV = os.getenv("FLASK_ENV")

class ProductionConfig(Config):
    uri = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")

    # Convert to SQLAlchemy-compatible URI if necessary
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

    SQLALCHEMY_DATABASE_URI = uri
    DEBUG = False


class DevelopmentConfig(Config):
    uri = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")

    # Convert to SQLAlchemy-compatible URI if necessary
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

    SQLALCHEMY_DATABASE_URI = uri
    DEBUG = False


# Optional: Environment-specific configuration loader for convenience
def get_config():
    env = os.getenv("FLASK_ENV", "development").lower()
    print(f'FLASK_ENV in get_config(): {env}')
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
