INTELLIGENT DATA SUITE

“When Data Talks, We Listen.”
<p align="center"> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"> <img src="https://img.shields.io/badge/ML-Sklearn-3776AB?style=flat-square&logo=scikit-learn&logoColor=white"> <img src="https://img.shields.io/badge/Visualization-Plotly-3C4F76?style=flat-square&logo=plotly&logoColor=white"> <img src="https://img.shields.io/badge/Voice-gTTS | SpeechRecognition-4DB6AC?style=flat-square"> </p> <p align="center"> <b>Empowering humans to explore, clean, and predict with data — using words, not code.</b> </p>
Overview

Intelligent Data Suite is a unified AI + Voice + Data Intelligence platform built with Streamlit.
It automates dataset assessment, cleaning, visualization, and prediction — all controlled through natural voice commands.

Vision

Data analysis should feel conversational, not procedural.
This suite transforms analysis into interaction through:

AI reasoning for natural query understanding

Voice-based recognition and response

Intelligent imputation and predictive modeling

Transparent, timestamped audit trails

A digital analyst that understands your data — and speaks back.

Key Modules
Module	Functionality	Core Logic
1. Upload & Assessment	Evaluates dataset quality (missing %, duplicates, outliers, score)	Weighted scoring algorithm visualized via Plotly Gauge
2. Cleaning & Imputation	Repairs missing data automatically	Mean, Median, or KNN-based imputation
3. Filtering & Selection	Provides flexible, persistent filtering	Multi-select logic with memory persistence
4. Visualization	Generates interactive charts	Bar, Line, Scatter, Boxplot, Histogram, Pie
5. AI Voice Queries	Executes spoken commands	SpeechRecognition + NLP-style parsing
6. Linear Prediction	Builds regression models dynamically	LinearRegression with live prediction overlay
7. Save/Load Sessions	Resumes prior sessions seamlessly	Pickle-based session persistence
8. Audit Trail	Logs every user action	JSON-based timestamped logging
System Architecture
User
 ├── Streamlit Interface
 │     ├── Upload & View Data
 │     ├── AI Query Engine <--> Voice System (SpeechRecognition + pyttsx3)
 │     ├── Data Processor (Pandas + NumPy + KNNImputer)
 │     ├── Visualization Core (Plotly)
 │     ├── Prediction Engine (LinearRegression)
 │     └── Audit Logger (JSON)

Intelligence Highlights

Data Quality IQ™ — Computes dataset integrity with weighted metrics for missing, duplicate, and outlier values.

Self-Aware Imputation — Selects Mean, Median, or KNN based on data type and density.

Adaptive Visualization — Automatically adjusts chart type based on column semantics.

Conversational AI Queries — Responds to natural speech or typed commands.

Immutable Audit Logging — Every step is timestamped and serialized for transparency.

Tech Stack
Layer	Technology
Frontend & UI	Streamlit
Data Wrangling	Pandas, NumPy
Visualization	Plotly Express, Plotly Graph Objects
Machine Learning	scikit-learn (Linear Regression, KNN Imputer)
Voice Intelligence	gTTS, pyttsx3, SpeechRecognition
Persistence	Pickle, JSON logs
Installation
# 1. Clone the repository
git clone https://github.com/<your-username>/Intelligent-Data-Suite.git
cd Intelligent-Data-Suite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py

requirements.txt
streamlit
pandas
numpy
plotly
scikit-learn
gTTS
pyttsx3
SpeechRecognition

Example Workflow

Use Case: Sales Dataset Analysis

Upload sales.csv

View the Data Quality Gauge

Apply KNN Imputation for missing values

Filter by Region = South Zone

Visualize a Bar Chart of Sales vs Product

Query: What is the mean of sales?

Build a Linear Regression model: Sales ~ AdSpend

Review the Audit Trail for all actions

Audit Trail Example
{
  "timestamp": "2025-08-19 18:42:07",
  "action": "imputation",
  "details": "Method: KNN"
}


Each operation is logged for governance and reproducibility.

Visualization Preview
<p align="center"> <img src="https://github.com/<your-username>/Intelligent-Data-Suite/assets/preview_dashboard.png" width="90%"> </p>

The Data Quality Gauge — the heartbeat of your dataset.

Future Enhancements

GPT-based semantic query understanding

Multivariate regression and classification modules

Voice-triggered visualization (e.g., "Show scatterplot of age vs income")

Cloud-synced audit trail (MongoDB + Streamlit Cloud)
