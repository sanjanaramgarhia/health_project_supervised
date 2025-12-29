# CardioPredict Pro

A machine learning-powered clinical decision support system for cardiovascular disease risk prediction using an ensemble of 8 ML algorithms.

## Features

- **📊 Data Analytics**: Comprehensive analysis of patient demographics and clinical parameters with interactive visualizations
- **💬 AI Chatbot**: Intelligent question-answering system for health-related queries
- **🎯 Risk Prediction**: Multi-model ensemble approach for accurate cardiovascular risk assessment
- **📈 Clinical Insights**: Detailed statistical analysis and patient demographics

## Technology Stack

This application utilizes an ensemble of 8 state-of-the-art machine learning algorithms:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Classifier (SVC)
- XGBoost
- CatBoost
- LightGBM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sanjanaramgarhia/health_project_Surpervised.git
cd health_project_Surpervised
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data (done automatically on first run)

4. Run the application:
```bash
streamlit run app.py
```

## Usage

The application will start on `http://localhost:8501`. Navigate through the tabs to:
- **Home**: Overview and introduction
- **Analysis**: View clinical data analytics and visualizations
- **Chatbot**: Ask health-related questions
- **Predict**: Enter patient parameters for cardiovascular risk assessment

## Disclaimer

This tool is designed to assist healthcare professionals and should not replace clinical judgment. All predictions are based on statistical models and should be interpreted in conjunction with clinical expertise and patient history.

## License

This project is for educational and research purposes.

