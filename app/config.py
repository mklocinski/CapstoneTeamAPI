import os

class Config:
    # Get the DATABASE_URL from the environment
    uri = os.getenv("DATABASE_URL")  # Heroku provides this environment variable

    # Replace the deprecated 'postgres://' with 'postgresql+psycopg2://'
    if uri and uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql+psycopg2://", 1)

    # Set SQLAlchemy database URI
    SQLALCHEMY_DATABASE_URI = uri

    # Other configurations
    SQLALCHEMY_TRACK_MODIFICATIONS = False
