# Patient Readmission Risk Predictor

Final Project for Applied Artificial Intelligence at Polytechnic of Santarém.

This project focuses on predicting 30-day hospital readmission risk for diabetic patients using the **Diabetes 130-US Hospitals** dataset. It includes data preprocessing, exploratory data analysis, machine learning modeling (Logistic Regression, Random Forest, XGBoost), deep learning modeling (Keras/TensorFlow), and an interactive web dashboard for visualization.

## Project Structure

* `dashboard.py`: Interactive Streamlit dashboard to explore the dataset, visualize data, and compare model performances.
* `main.ipynb`: Jupyter Notebook containing the detailed data analysis, preprocessing steps, and model training processes.
* `requirements.txt`: List of Python dependencies required to run the project.
* `data/`: Directory containing the datasets (`diabetic_data.csv`, `IDS_mapping.csv`).
* `models/`: Directory where trained models are saved (`logistic_regression.joblib`, `random_forest.joblib`, `xgboost.joblib`, `keras_model.h5`).

## Setup Instructions

Follow these steps to set up the project environment and run the applications.

### 1. Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### 2. Create and Activate the Environment

It is recommended to use a dedicated Conda environment with Python 3.10.

```bash
# Create the environment
conda create -n readmission-env python=3.10 -y

# Activate the environment
conda activate readmission-env
```

### 3. Install Dependencies

Once the environment is active, install the required packages using `pip` and the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*(Key dependencies include: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, and `tensorflow`.)*

## Running the Project

### Interactive Dashboard

To launch the interactive web dashboard, run the following command from the root directory of the project:

```bash
streamlit run dashboard.py
```

This will start a local web server and open the dashboard in your default web browser (typically at `http://localhost:8501`). The dashboard allows you to:
* View a cleaned summary of the dataset.
* Explore data distributions and correlations (EDA).
* Compare the performance of the trained machine learning and deep learning models (Accuracy, ROC-AUC, Feature Importance).

### Analysis Notebook

To view or rerun the detailed data analysis and model training steps, start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook main.ipynb
# or
jupyter lab main.ipynb
```

This notebook contains the step-by-step pipeline for loading the raw data, engineering features, and training the models that are used by the dashboard.
