import os

class Config:
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ProductionConfig(Config):
    # Use DATABASE_URL for Heroku production, which is automatically set by Heroku
    uri = os.getenv("DATABASE_URL")
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)
    SQLALCHEMY_DATABASE_URI = uri

class DevelopmentConfig(Config):
    # Local development database configuration (SQLite)
    SQLALCHEMY_DATABASE_URI = "sqlite:///local.db"
    DEBUG = True
