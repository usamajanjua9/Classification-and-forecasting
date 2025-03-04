import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# Customizing Streamlit UI
st.set_page_config(page_title="ML Detection & Forecasting App", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .css-1rs6os {visibility: hidden;}
    .reportview-container {background-color: #f5f5f5;}
    .sidebar .sidebar-content {background-color: #263238; color: white;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stTextInput>div>div>input {border-radius: 5px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Sidebar
st.title("📊 ML Model: Detection & Forecasting")
st.sidebar.header("🔧 Options")
option = st.sidebar.radio("Select Task:", ("Detection (Classification)", "Forecasting (Time Series)"))

if option == "Detection (Classification)":
    st.header("🔍 Classification Model")
    
    # Load Example Dataset
    def load_classification_data():
        from sklearn.datasets import load_iris, load_wine, load_digits
        dataset_choice = st.sidebar.selectbox("Choose Dataset:", ["Iris", "Wine", "Digits"])
        if dataset_choice == "Iris":
            data = load_iris()
        elif dataset_choice == "Wine":
            data = load_wine()
        else:
            data = load_digits()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    
    df = load_classification_data()
    st.subheader("📌 Example Dataset")
    st.dataframe(df.head())
    
    # Splitting Data
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_choice = st.sidebar.selectbox("🛠 Choose Model:", ["Random Forest", "Gradient Boosting", "AdaBoost", "Support Vector Machine"])
    
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_choice == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
    else:
        model = SVC(probability=True, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Display Metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("📊 Model Performance")
    st.write(f"**✔ Accuracy:** {round(accuracy * 100, 2)} %")
    
    st.subheader("📜 Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    st.subheader("🗂 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)
    
    # ROC Curve
    if y_proba is not None:
        st.subheader("📈 ROC Curve")
        fig, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label=y_test.max())
        ax.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)
    
    # User Prediction
    st.subheader("📝 Predict New Data")
    user_input = [st.number_input(col, value=0.0) for col in X.columns]
    if st.button("Predict"):
        pred = model.predict([user_input])[0]
        st.success(f"🔮 Predicted Class: {pred}")

elif option == "Forecasting (Time Series)":
    st.header("📈 Time Series Forecasting")
    
    # Load Example Time Series Data
    def load_forecasting_data()
        dates = pd.date_range(start="2022-01-01", periods=50, freq='D')
        values = np.cumsum(np.random.randn(50) * 10)
        df = pd.DataFrame({'Date': dates, 'Value': values})
        return df
    
    df = load_forecasting_data()
    st.subheader("📌 Example Dataset")
    st.line_chart(df.set_index("Date"))
    
    # Model Selection
    forecast_model = st.sidebar.selectbox("🛠 Select Forecasting Model:", ["ARIMA", "Exponential Smoothing"])
    
    if forecast_model == "ARIMA":
        model = ARIMA(df['Value'], order=(5, 1, 0))
    else:
        model = ExponentialSmoothing(df['Value'], trend='add', seasonal='add', seasonal_periods=7)
    
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    
    # Display Forecast
    st.subheader("📊 Forecasted Values")
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10, freq='D')
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    st.line_chart(forecast_df.set_index("Date"))
    st.dataframe(forecast_df)
    
    st.success("✔ Forecasting using the selected model is completed.")

st.sidebar.info("🚀 Developed by Dr. Usama Arshad")
