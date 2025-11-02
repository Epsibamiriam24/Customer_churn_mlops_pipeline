# Customer Churn Prediction MLOps Project

## Project Overview
This project implements an end-to-end MLOps pipeline for customer churn prediction, including:
- Automated data preprocessing
- Model training and evaluation
- CI/CD pipeline
- Docker containerization
- Streamlit web interface

## Project Structure
```
Customer-churn/
├── src/                    # Source code
│   ├── app/               # Streamlit application
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction script
│   ├── model.py          # Model definition
│   └── utils.py          # Utility functions
├── tests/                 # Unit tests
├── Data/                  # Data files
├── artifacts/             # Model artifacts
├── mlruns/               # MLflow tracking
└── docker/               # Docker configuration
```

## Getting Started

### Prerequisites
- Python 3.9+
- Docker
- Git

### Installation
1. Clone the repository:
```bash
git clone https://github.com/epsibamiriam/Customer-churn.git
cd Customer-churn
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Local Development
1. Train the model:
```bash
python src/train.py --train Data/train.csv --test Data/test.csv
```

2. Start the Streamlit app:
```bash
streamlit run src/app/streamlit_app.py
```

#### Using Docker
1. Build the container:
```bash
docker build -t customer-churn .
```

2. Run the container:
```bash
docker run -p 8501:8501 customer-churn
```

## CI/CD Pipeline
The project includes GitHub Actions workflows for:
- Automated testing
- Code quality checks
- Model training
- Docker image building
- Deployment

## Testing
Run the tests with:
```bash
pytest tests/ --cov=src
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.