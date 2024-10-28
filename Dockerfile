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
RUN curl -O https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz \
    && tar -xvf Python-3.7.12.tgz \
    && cd Python-3.7.12 \
    && ./configure --enable-optimizations --with-ssl \
    && make -j8 \
    && make altinstall \
    && make test || true  # Skip test errors

# Set the working directory
WORKDIR /app

# Set PYTHONPATH to include /app directory
ENV PYTHONPATH=/app
ENV FLASK_APP=app
ENV FLASK_ENV=production

# Copy and install main API dependencies for Python 3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "import pandas; print('Pandas loaded successfully in main env:', pandas.__version__)"

# Copy the rest of the application code
COPY . .

# Create and set up the Python 3.7 virtual environment for model dependencies
RUN /usr/local/bin/python3.7 -m venv /app/model_venv
RUN /app/model_venv/bin/pip install psycopg2-binary
RUN /app/model_venv/bin/pip install pandas numpy flask_sqlalchemy flask_migrate python-dotenv
RUN /app/model_venv/bin/pip install ./models/drlss  # Install model dependencies
RUN /app/model_venv/bin/pip install protobuf==3.20.3

# Confirm pandas installation in 3.7 environment
RUN /app/model_venv/bin/python -c "import pandas; print('Pandas version in model venv:', pandas.__version__)"

# Expose port 8000 for the API (Heroku dynamically sets the actual port)
EXPOSE 8000

# Command to run migrations and start the API
CMD flask db upgrade && exec gunicorn app:app --bind 0.0.0.0:${PORT}