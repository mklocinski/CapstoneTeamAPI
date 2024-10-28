import os


class Config:
    """Base configuration with default settings shared across environments."""
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disables modification tracking to save memory


class ProductionConfig(Config):
    """Production configuration used on Heroku."""
    # Attempt to get SQLALCHEMY_DATABASE_URI; fall back to DATABASE_URL if necessary
    uri = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")

    # Convert to SQLAlchemy-compatible URI if necessary
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

    SQLALCHEMY_DATABASE_URI = uri
    DEBUG = False  # Production should generally have DEBUG off


class DevelopmentConfig(Config):
    """Development configuration for local testing."""
    # SQLite database for quick local development
    SQLALCHEMY_DATABASE_URI = "sqlite:///local.db"
    DEBUG = True  # Enables debug mode for easier troubleshooting


# Optional: Environment-specific configuration loader for convenience
def get_config():
    """Returns the appropriate config class based on the FLASK_ENV variable."""
    env = os.getenv("FLASK_ENV", "development").lower()
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
