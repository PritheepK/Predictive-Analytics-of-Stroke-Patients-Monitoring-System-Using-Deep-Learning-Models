# Predictive Analytics of Stroke Patients Monitoring System Using Deep Learning Models

This project is a full-stack healthcare analytics solution that leverages deep learning to detect various heart conditions from ECG images. It features a user-friendly Flask web application for uploading ECG scans, performing real-time predictions, storing patient records, and generating PDF medical reports.

---

## Features

- Deep Learning Model (ConvNeXt) trained to classify:
  - Atrial Fibrillation (AFib)
  - Cardiomegaly (Enlarged Heart)
  - Myocardial Infarction (Heart Attack)
  - Arrhythmia-related Heart Block
  - Normal

- Flask Web App with:
  - User registration & login (SQLite)
  - Patient data input form
  - Image upload & prediction
  - PDF report generation
  - Patient record saving (Excel)

- Model Performance:
  - Accuracy: ~86% on validation data
  - Evaluation: Precision, Recall, F1-score (stored in classification_report.txt)

---

## Technologies Used

- Python
- PyTorch + timm (ConvNeXt) – Model training & inference
- Flask – Web framework
- SQLite – User authentication
- OpenPyXL – Excel data storage
- ReportLab – PDF report generation
- HTML/CSS – Frontend templates

--