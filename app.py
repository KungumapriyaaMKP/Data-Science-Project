
import streamlit as st
st.set_page_config(
    page_title="Intelligent Data Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer # for imputation
from sklearn.linear_model import LinearRegression # ""
import plotly.express as px # for plots graphs
import plotly.graph_objects as go 
import pickle # to save and load sessions , also to maintain audit logs
import json
from datetime import datetime # just simple timestamps

# voice features
voice_enabled = False
try:
    from gtts import gTTS # tts
    import pyttsx3
    import speech_recognition as sr
    voice_enabled = True
except:
    st.warning("Voice features disabled. Install gTTS, pyttsx3, SpeechRecognition for voice support.")

# Title

st.title("Intelligent Data Suite With Voice query")
st.markdown("---")

# Sidebar Navigation
step = st.sidebar.radio("Navigation", [
    "1. Upload & Assessment",
    "2. Cleaning & Imputation",
    "3. Filtering & Selection",
    "4. Visualization",
    "5. AI Queries",
    "6. Prediction",
    "7. Save/Load Session",
    "8. Audit Trail"
])

# Session State Initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Helper Functions
#audit log maintaining
def log_action(action, details=""):
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "action": action, "details": details}
    st.session_state.logs.append(entry)
    with open("audit_log.json","a") as f:
        f.write(json.dumps(entry)+"\n")
        
#Data quality calculation
def calculate_quality(df):
    total_cols = df.shape[1]
    total_rows = df.shape[0]
    
    
#per col sum + all series add (finally) => within each col + across all col
    missing_pct = df.isnull().sum().sum() / (total_rows*total_cols) * 100
    duplicate_pct = df.duplicated().sum() / total_rows * 100

    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        z = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std(ddof=0))
        outlier_pct = (z > 3).sum().sum() / np.prod(numeric_cols.shape) * 100
    else:
        outlier_pct = 0

    total_quality = 100 - (missing_pct*0.5 + duplicate_pct*0.2 + outlier_pct*0.3)
    return round(total_quality,2), round(missing_pct,2), round(duplicate_pct,2), round(outlier_pct,2)

# tto -> text to output
def speak_text(text):
    if voice_enabled:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

# 1. Upload & Assessment
if step == "1. Upload & Assessment":
    st.header("Upload Dataset & Assess Quality")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv','xlsx'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df.copy()
        st.session_state.df_filtered = df.copy()
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
#display all calculated values
        score, miss, dup, outl = calculate_quality(df)
        st.subheader("Data Quality Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Quality", f"{score}%")
        col2.metric("Missing %", f"{miss}%")
        col3.metric("Duplicates %", f"{dup}%")
        col4.metric("Outliers %", f"{outl}%")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Data Quality Score"},
            gauge={'axis': {'range':[0,100]},
                   'bar': {'color':'green'},
                   'steps':[{'range':[0,50],'color':'red'},
                            {'range':[50,75],'color':'yellow'},
                            {'range':[75,100],'color':'green'}]}))
        st.plotly_chart(fig, use_container_width=True)
        log_action("upload_dataset", uploaded_file.name)

# 2. Cleaning & Imputation
elif step == "2. Cleaning & Imputation":
    st.header("Data Cleaning & Imputation")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        score, miss, dup, outl = calculate_quality(df)
        st.subheader("Current Data Quality")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Quality", f"{score}%")
        col2.metric("Missing %", f"{miss}%")
        col3.metric("Duplicates %", f"{dup}%")
        col4.metric("Outliers %", f"{outl}%")

        if st.checkbox("Do you want to impute missing data?"):
            method = st.selectbox("Choose Imputation Method", ["Mean", "Median", "KNN"])
            if st.button("Apply Imputation"):
                numeric_cols = df.select_dtypes(include=np.number).columns
                cat_cols = df.select_dtypes(include='object').columns

                # Capture missing BEFORE imputation
                missing_before = df.isnull().sum()

                # Numeric imputation
                if method == "Mean":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif method == "Median":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                else:
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

                # Categorical imputation
                for col in cat_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])

                st.session_state.df = df.copy()
                st.session_state.df_filtered = df.copy()
                st.success("Imputation Applied Successfully")
                log_action("imputation", method)

                # Updated quality score 
                new_score, new_miss, new_dup, new_outl = calculate_quality(df)
                col1.metric("Updated Quality", f"{new_score}%")
                col2.metric("Updated Missing %", f"{new_miss}%")
                col3.metric("Updated Duplicates %", f"{new_dup}%")
                col4.metric("Updated Outliers %", f"{new_outl}%")

                # Missing values before vs after - bar chart
                missing_after = df.isnull().sum()
                missing_df = pd.DataFrame({
                    "Column": df.columns,
                    "Before": missing_before,
                    "After": missing_after
                })
                fig2 = px.bar(missing_df, x="Column", y=["Before","After"],
                              barmode="group", title="Missing Values Before vs After")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Upload a dataset first!")

# 3. Filtering & Selection
elif step == "3. Filtering & Selection":
    st.header("Filter Data for Analysis")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        filter_col = st.selectbox("Select Column to Filter (Optional)", [None]+list(df.columns))
        if filter_col:
            unique_values = df[filter_col].dropna().unique()
            selected_values = st.multiselect(f"Select value(s) from {filter_col}", unique_values)
            if selected_values:
                df_filtered = df[df[filter_col].isin(selected_values)]
            else:
                df_filtered = df.copy()
        else:
            df_filtered = df.copy()
        st.session_state.df_filtered = df_filtered
        st.subheader("Filtered Dataset Preview")
        st.dataframe(df_filtered.head())
        log_action("filter_data", f"Filtered {filter_col} with {selected_values if filter_col else 'All'}")
    else:
        st.warning("Upload a dataset first!")

# 4️⃣ Visualization
elif step == "4. Visualization":
    st.header("Interactive Visualization")
    if st.session_state.df_filtered is not None:
        df_filtered = st.session_state.df_filtered
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_filtered.select_dtypes(include='object').columns.tolist()

        st.subheader("Numeric Columns Visualization")
        if numeric_cols:
            col1, col2 = st.columns([3,1])
            with col2:
                chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Boxplot", "Histogram"])
                x_axis = st.selectbox("X-axis", df_filtered.columns)
                y_axis = st.selectbox("Y-axis", numeric_cols)
            with col1:
                if chart_type == "Bar":
                    fig = px.bar(df_filtered, x=x_axis, y=y_axis)
                elif chart_type == "Line":
                    fig = px.line(df_filtered, x=x_axis, y=y_axis)
                elif chart_type == "Scatter":
                    fig = px.scatter(df_filtered, x=x_axis, y=y_axis)
                elif chart_type == "Boxplot":
                    fig = px.box(df_filtered, x=x_axis, y=y_axis)
                else:
                    fig = px.histogram(df_filtered, x=y_axis)
                st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:
            st.subheader("Categorical Columns Distribution")
            cat_col = st.selectbox("Choose Categorical Column", categorical_cols)
            fig_cat = px.pie(df_filtered, names=cat_col, title=f"Distribution of {cat_col}")
            st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.warning("Filter or upload dataset first!")

# 5. AI Queries
elif step == "5. AI Queries":
    st.header("Natural Language Data Queries")
    if st.session_state.df_filtered is not None:
        df_filtered = st.session_state.df_filtered
        query = st.text_input("Ask a question (e.g., mean of column, list brands)")
        if query:
            q = query.lower()

            response = None
            # Numeric queries
            if "mean" in q:
                response = df_filtered.mean(numeric_only=True)
            elif "sum" in q:
                response = df_filtered.sum(numeric_only=True)
            elif "describe" in q:
                response = df_filtered.describe(include='all')
            # Column-specific queries
            else:
                for col in df_filtered.columns:
                    if col.lower() in q:
                        response = df_filtered[col].unique()
                        break
            if response is None:
                response = "Query not recognized. Try: mean, sum, describe, or a column name"
            st.write(response)
            speak_text(str(response))
            log_action("AI_query", query)
    else:
        st.warning("Filter or upload dataset first!")

# 6. Prediction using Linear REgression
elif step == "6. Prediction":
    st.header("Run Prediction (Linear Regression)")
    if st.session_state.df_filtered is not None:
        df_filtered = st.session_state.df_filtered
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            feature = st.selectbox("Select Feature Column", numeric_cols)
            target = st.selectbox("Select Target Column", [col for col in numeric_cols if col != feature])
            if st.button("Run Prediction"):
                X = df_filtered[[feature]]
                y = df_filtered[target]
                model = LinearRegression()
                model.fit(X, y)
                df_filtered['Prediction'] = model.predict(X)
                st.session_state.df_filtered = df_filtered

                fig_pred = px.scatter(df_filtered, x=feature, y=[target,'Prediction'], title=f"{target} vs Prediction")
                st.plotly_chart(fig_pred, use_container_width=True)
                st.success("Prediction Completed")
                log_action("prediction", f"Feature: {feature}, Target: {target}")
        else:
            st.write("Need at least 2 numeric columns for prediction")
    else:
        st.warning("Filter or upload dataset first!")

# 7. Save/Load Session
elif step == "7. Save/Load Session":
    st.header("Save or Load Analysis Session")
    if st.session_state.df_filtered is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Session"):
                with open("saved_session.pkl", "wb") as f:
                    pickle.dump(st.session_state.df_filtered, f)
                st.success("Session Saved")
                log_action("save_session")
        with col2:
            if st.button("Load Session"):
                try:
                    with open("saved_session.pkl","rb") as f:
                        df_loaded = pickle.load(f)
                    st.session_state.df_filtered = df_loaded
                    st.dataframe(df_loaded.head())
                    st.success("Session Loaded")
                    log_action("load_session")
                except:
                    st.error("No saved session found")
    else:
        st.warning("Filter or upload dataset first!")

# 8. Audit Trail
elif step == "8. Audit Trail":
    st.header("Audit Trail")
    if st.session_state.logs:
        for entry in st.session_state.logs[::-1]:
            st.markdown(f"[{entry['timestamp']}] {entry['action']}: {entry['details']}")
    else:
        st.write("No actions logged yet.")
