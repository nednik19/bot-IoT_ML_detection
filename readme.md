# IoT Botnet Anomaly Detection

This project focuses on building and evaluating machine learning models for detecting anomalies in IoT network traffic, which are indicative of potential botnet activity. The project uses the UNSW 2018 IoT Botnet Dataset for training and evaluation.

## Features
- **Data Preprocessing:** Efficient handling of large datasets, including sampling and scaling.
- **Anomaly Detection Models:** Implementation of models like Random Forest, Isolation Forest, and One-Class SVM.
- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, and others are calculated to assess model performance.

---

## Folder Structure
- **`csv_anomaly_detection.py`:** Python script for preprocessing and sampling the dataset.
- **`train_model.ipynb`:** Jupyter Notebook for training, testing, and evaluating machine learning models.
- **`saved_model/`:** Directory for storing trained models and preprocessing scalers.

---

## Dataset
The UNSW 2018 IoT Botnet Dataset is used in this project. It contains network traffic data from IoT devices under various normal and attack scenarios.

- **Download Link:** [UNSW IoT Botnet Dataset](https://research.unsw.edu.au/projects/bot-iot-dataset)

---

## Requirements
Install the required Python packages using the following command:

```bash
pip install -r requirements.txt
