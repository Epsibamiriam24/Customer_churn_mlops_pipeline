FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ src/
COPY Data/ Data/

# Create artifacts directory and train the model using the correct training script
RUN mkdir -p artifacts && \
    echo "Training model with correct features..." && \
    python src/train.py --train Data/clean_train.csv --test Data/clean_test.csv --out artifacts/model.joblib

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Expose the port
EXPOSE 8501

# Command to run the Streamlit app (using JSON array format for better signal handling)
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
