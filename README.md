INTELLIGENT DATA SUITE - “When Data Talks, We Listen.” 🎙️

AI + Voice + Data Intelligence — A unified Streamlit platform for automated data analysis, intelligent cleaning, visualization, and prediction — all controllable via natural voice commands.

<p align="center"> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"> <img src="https://img.shields.io/badge/ML-Sklearn-3776AB?style=flat-square&logo=scikit-learn&logoColor=white"> <img src="https://img.shields.io/badge/Visualization-Plotly-3C4F76?style=flat-square&logo=plotly&logoColor=white"> <img src="https://img.shields.io/badge/Voice-gTTS | SpeechRecognition-4DB6AC?style=flat-square"> </p> <p align="center"> <b>Empowering humans to explore, clean, and predict with data — using words, not code.</b> </p>
🧭 Vision

Data analysis shouldn’t feel like programming — it should feel like a conversation.
Intelligent Data Suite bridges that gap by combining:

AI reasoning for interpreting queries

Speech recognition & synthesis for true voice interactivity

Smart data imputation & prediction models

Live audit tracking for total transparency

A digital analyst that understands your data — and talks back.

🌌 Key Modules
🚀 Module	🧠 What It Does	💡 Magic Behind It
1️⃣ Upload & Assessment	Instantly grades your dataset’s health (Missing %, Duplicates, Outliers, Quality Score)	Uses a custom weighted scoring algorithm visualized through a dynamic Plotly Gauge
2️⃣ Cleaning & Imputation	Automatically repairs missing data	Choose from Mean, Median, or KNN Imputation (adaptive ML-based filling)
3️⃣ Filtering & Selection	Filter large datasets interactively	Multi-select logic + memory persistence
4️⃣ Visualization	Generates smart, interactive charts	Bar, Line, Scatter, Boxplot, Histogram, and Pie – all Plotly powered
5️⃣ AI Voice Queries	Talk to your data!	Understands “mean of sales”, “describe data”, “list brands”, etc.
6️⃣ Linear Prediction	Creates regression models in one click	Real-time training + prediction overlay
7️⃣ Save/Load Sessions	Resume your analysis anytime	Powered by pickle state serialization
8️⃣ Audit Trail	Every action is logged	Time-stamped entries written to JSON audit log
🧩 Architecture Overview
User → Streamlit UI
        ├── Upload & View Data
        ├── AI Query Engine ←→ Voice System (SpeechRecognition + pyttsx3)
        ├── Data Processor (Pandas + NumPy + KNNImputer)
        ├── Visualization Core (Plotly)
        ├── Prediction Model (LinearRegression)
        └── Audit Layer (JSON Log)


💬 Result: A modular intelligence layer that lets users “converse” with their data.

🧠 Smart Intelligence Highlights

✨ Data Quality IQ™ - Computes dataset integrity with weighted penalties for missing, duplicate, and outlier ratios.
🎯 Self-Aware Imputation - Knows when to apply KNN vs Mean/Median.
🎨 Adaptive Visualization - Changes chart types dynamically based on column types.
🗣️ Conversational AI Queries - Talk, type, or combine both.
🔐 Immutable Audit Logging - Each user interaction is timestamped and serialized to JSON.

🛠️ Tech Stack
Layer	Technology
Frontend & UI	Streamlit
Data Wrangling	Pandas, NumPy
Visualization	Plotly Express, Graph Objects
Machine Learning	scikit-learn (Linear Regression, KNN Imputer)
Voice Intelligence	gTTS, pyttsx3, SpeechRecognition
Persistence	Pickle + JSON logs
🧩 Installation & Run
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/Intelligent-Data-Suite.git
cd Intelligent-Data-Suite

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run app.py


📦 requirements.txt

streamlit
pandas
numpy
plotly
scikit-learn
gTTS
pyttsx3
SpeechRecognition

🧬 Example Workflow

💡 "Let’s find out how healthy my sales dataset is..."

Step 1: Upload → sales.csv
Step 2: Check your Data Quality Gauge
Step 3: Apply KNN Imputation to fix missing values
Step 4: Filter by region → South Zone
Step 5: Visualize → Bar chart of Sales vs Product
Step 6: Ask: “What’s the mean of sales?”
Step 7: Run Linear Regression for Sales ~ AdSpend
Step 8: Review Audit Trail to see every action you performed

And yes… it can speak the answers back 🗣️

🧾 Audit Trail Example
{
  "timestamp": "2025-08-19 18:42:07",
  "action": "imputation",
  "details": "Method: KNN"
}


Every dataset touchpoint is transparently recorded for governance and reproducibility.

**Visuals**
<p align="center"> <img src="https://github.com/<your-username>/Intelligent-Data-Suite/assets/preview_dashboard.png" width="90%"> </p>

The Data Quality Gauge — the heartbeat of your dataset.

💡 Future Enhancements

✅ GPT-based semantic query understanding
✅ Multivariate regression and classification modules
✅ Voice command triggers for visualization (“Show me a scatterplot of age vs income”)
✅ Cloud audit sync (MongoDB + Streamlit Cloud)
