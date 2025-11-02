FROM python:3.9-slim as builder

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

# Copy the entire application including artifacts
COPY . .

# Ensure artifacts directory exists and train initial model if needed
RUN mkdir -p artifacts && \
    if [ ! -f artifacts/model.joblib ]; then \
        echo "Training initial model..." && \
        python src/initial_model.py; \
    fi

# Set environment variables
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Expose the port
EXPOSE 8501

# Command to run the Streamlit app
CMD streamlit run src/app/streamlit_app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS true \
    --server.enableXsrfProtection true
