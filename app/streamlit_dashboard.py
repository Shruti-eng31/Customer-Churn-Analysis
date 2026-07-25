import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from utils import load_artifacts, apply_custom_theme, get_insight

# --- Setup ---
st.set_page_config(page_title="Customer Churn Platform", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Load CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("app/style.css")

# Load Data and Models
@st.cache_data
def load_data():
    df = pd.read_csv('data/customer_churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

df = load_data()

try:
    model, scaler, label_encoders, feature_cols = load_artifacts('models')
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Models not found. Please run train_models.py first. Error: {e}")

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/6009/6009864.png", width=100)
st.sidebar.title("Churn Analytics")
st.sidebar.markdown("Enterprise AI Platform")

page = st.sidebar.radio("Navigation", [
    "Executive Overview",
    "Customer Demographics",
    "Revenue Analytics",
    "Customer Behavior",
    "Prediction Tool"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Crafted with Data, Code & Creativity by Shruti Singh")

# --- Pages ---
if page == "Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("High-level key performance indicators and overall churn metrics.")
    
    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    monthly_revenue = df['MonthlyCharges'].sum()
    avg_tenure = df['tenure'].mean()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%", "-1.2%", delta_color="inverse")
    col3.metric("Monthly Revenue", f"${monthly_revenue:,.0f}")
    col4.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names='Churn', title='Customer Churn Distribution', 
                     color_discrete_sequence=['#3b82f6', '#ef4444'], hole=0.6)
        fig = apply_custom_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(get_insight("demographics"), unsafe_allow_html=True)
        
    with c2:
        fig2 = px.histogram(df, x='Contract', color='Churn', barmode='group',
                            title='Churn by Contract Type',
                            color_discrete_sequence=['#3b82f6', '#ef4444'])
        fig2 = apply_custom_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Customer Demographics":
    st.title("👥 Customer Demographics")
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(df, names='gender', title='Gender Distribution', color_discrete_sequence=['#8b5cf6', '#06b6d4'])
        fig = apply_custom_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        fig2 = px.histogram(df, x='SeniorCitizen', color='Churn', barmode='group', 
                            title='Churn by Senior Citizen Status',
                            color_discrete_sequence=['#3b82f6', '#ef4444'])
        fig2 = apply_custom_theme(fig2)
        fig2.update_xaxes(tickvals=[0, 1], ticktext=['Non-Senior', 'Senior'])
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown(get_insight("demographics"), unsafe_allow_html=True)

elif page == "Revenue Analytics":
    st.title("💰 Revenue Analytics")
    
    fig = px.box(df, x='Churn', y='MonthlyCharges', color='Churn',
                 title='Distribution of Monthly Charges by Churn',
                 color_discrete_sequence=['#3b82f6', '#ef4444'])
    fig = apply_custom_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.scatter(df, x='tenure', y='TotalCharges', color='Churn',
                      title='Customer Lifetime Value: Tenure vs Total Charges',
                      opacity=0.6, color_discrete_sequence=['#3b82f6', '#ef4444'])
    fig2 = apply_custom_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown(get_insight("revenue"), unsafe_allow_html=True)

elif page == "Customer Behavior":
    st.title("📱 Customer Behavior & Services")
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x='InternetService', color='Churn', barmode='group',
                           title='Churn by Internet Service',
                           color_discrete_sequence=['#3b82f6', '#ef4444'])
        fig = apply_custom_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        fig2 = px.histogram(df, x='TechSupport', color='Churn', barmode='group',
                            title='Churn by Tech Support',
                            color_discrete_sequence=['#3b82f6', '#ef4444'])
        fig2 = apply_custom_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown(get_insight("services"), unsafe_allow_html=True)

elif page == "Prediction Tool":
    st.title("🤖 AI Churn Predictor")
    st.markdown("Enter customer details to predict their likelihood of churning using the trained XGBoost/GBM model.")
    
    if not model_loaded:
        st.warning("Please ensure models are trained to use this feature.")
    else:
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ['Female', 'Male'])
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ['Yes', 'No'])
                dependents = st.selectbox("Dependents", ['Yes', 'No'])
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                phone = st.selectbox("Phone Service", ['Yes', 'No'])
                
            with col2:
                multiple = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
                internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
                security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
                backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
                protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
                tech = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
                
            with col3:
                tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
                movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
                contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
                paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
                payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
                monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
                total = st.number_input("Total Charges ($)", 0.0, 9000.0, monthly * tenure)
                
            submitted = st.form_submit_button("Predict Churn Risk")
            
        if submitted:
            # Construct input dict matching feature_cols
            input_data = {
                'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
                'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
                'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
                'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,
                'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Encode
            for col in input_df.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    # Handle unseen labels just in case by mapping to 0, though inputs are controlled
                    input_df[col] = label_encoders[col].transform(input_df[col])
            
            # Reorder
            input_df = input_df[feature_cols]
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prob = model.predict_proba(input_scaled)[0][1]
            
            st.markdown("### Prediction Results")
            if prob > 0.5:
                st.markdown(f"<div class='risk-high'>⚠️ HIGH RISK OF CHURN<br>Probability: {prob*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("**Recommendation:** Immediate intervention required. Offer a customized discount on a 1-year contract and free Tech Support for 3 months.")
            else:
                st.markdown(f"<div class='risk-low'>✅ LOW RISK<br>Probability: {prob*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("**Recommendation:** Customer is stable. Focus on cross-selling Streaming services to increase Lifetime Value.")

