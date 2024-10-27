# Use Python 3.10 as the base for the API (compatible with numpy and pandas versions specified)
FROM python:3.10-slim

# Install system dependencies, including libraries needed for Python and psycopg2
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
    && rm -rf /var/lib/apt/lists/*

# Optional: Only if still needed for a separate environment model, download and install Python 3.7
RUN curl -O https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz \
    && tar -xvf Python-3.7.12.tgz \
    && cd Python-3.7.12 \
    && ./configure --enable-optimizations --with-ssl \
    && make -j8 \
    && make altinstall \
    && make test || true  # Skip test errors

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies for the API using Python 3.10
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Optional: If needed, create the Python 3.7 virtual environment for model
RUN /usr/local/bin/python3.7 -m venv /app/model_venv

# Activate the Python 3.7 virtual environment and install psycopg2-binary for PostgreSQL if needed
RUN /app/model_venv/bin/pip install --no-cache-dir psycopg2-binary

# Install other model dependencies inside the Python 3.7 virtual environment if required
RUN /app/model_venv/bin/pip install --no-cache-dir ./models/drlss

# Expose port 8000 for the API
EXPOSE 8000

# Command to run the API
CMD exec gunicorn app:app --bind 0.0.0.0:${PORT}
