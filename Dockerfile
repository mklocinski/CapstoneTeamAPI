# Use Python 3.10 as the base for the API
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    gcc \
    libopenmpi-dev \
    openmpi-bin \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    curl \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.7 for the model environment
RUN curl -O --retry 5 --retry-delay 10 https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz
RUN tar -xvf Python-3.7.12.tgz
RUN cd Python-3.7.12 && ./configure --enable-optimizations --with-ssl
RUN cd Python-3.7.12 && make -j$(nproc)
RUN cd Python-3.7.12 && make altinstall

# Set the working directory to the project root
WORKDIR /CapstoneTeamAPI

# Set environment variables
ENV PYTHONPATH=/CapstoneTeamAPI
ENV FLASK_APP=app
ENV FLASK_ENV=production
ENV PATH="/CapstoneTeamAPI/.venv_main/bin:$PATH"

# Copy the application files
COPY . .

# Create main app virtual environment with Python 3.10
RUN python3.10 -m venv /CapstoneTeamAPI/.venv_main
RUN /CapstoneTeamAPI/.venv_main/bin/pip install -r requirements.txt

# Create model-specific virtual environment with Python 3.7
RUN python3.7 -m venv /CapstoneTeamAPI/.venv_model
RUN /CapstoneTeamAPI/.venv_model/bin/pip install -e ./models/drlss

# Additional packages specific to the model environment
RUN /CapstoneTeamAPI/.venv_model/bin/pip install pandas==1.1.5 psycopg2-binary flask_sqlalchemy flask_migrate python-dotenv
RUN /CapstoneTeamAPI/.venv_model/bin/pip install protobuf==3.20.3

# Verify installations in both environments
RUN /CapstoneTeamAPI/.venv_main/bin/python -c "import numpy; print('Main app numpy version:', numpy.__version__)"
RUN /CapstoneTeamAPI/.venv_main/bin/python -c "import pandas; print('Pandas in main app:', pandas.__version__)"
RUN /CapstoneTeamAPI/.venv_model/bin/python -c "import numpy; print('Model env numpy version:', numpy.__version__)"
RUN /CapstoneTeamAPI/.venv_model/bin/python -c "import pandas; print('Model env pandas version:', pandas.__version__)"

# Expose port 8000 for the API (Heroku dynamically sets the actual port)
EXPOSE 8000

# Command to run migrations and start the API using the main app environment
CMD /CapstoneTeamAPI/.venv_main/bin/flask db upgrade && \
    exec /CapstoneTeamAPI/.venv_main/bin/gunicorn app:app --bind 0.0.0.0:${PORT}
