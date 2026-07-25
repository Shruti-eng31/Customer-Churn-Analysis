# Customer Churn Analysis & Prediction Platform 📊

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-17614B?logo=xgboost&logoColor=white)

A premium, end-to-end Data Science portfolio project that simulates a real-world enterprise analytics platform. This project is designed to analyze customer behavior, identify churn patterns, predict customer churn using Machine Learning, and provide actionable business insights through an interactive dashboard.

## 🚀 Features

- **Data Cleaning & Feature Engineering:** Automated pipeline to handle missing values and encode categorical features.
- **Exploratory Data Analysis (EDA):** Jupyter Notebook containing over 35 professional analyses using Plotly and Seaborn.
- **Machine Learning Models:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, and KNN.
- **AI Churn Predictor:** Predicts churn probability for individual customers and outputs risk categories (High/Low Risk).
- **Executive Dashboard:** A visually stunning Streamlit dashboard featuring:
  - Dark mode with Blue + Purple + Cyan gradients.
  - Glassmorphism & rounded cards.
  - Interactive Plotly visualizations.
  - Business intelligence and insight generators.

---

## 📂 Project Structure

```
Customer-Churn-Analysis/
├── app/
│   ├── streamlit_dashboard.py # Main Streamlit application
│   ├── style.css              # Custom CSS for the dashboard
│   └── utils.py               # Helper functions
├── data/                      # Dataset (downloaded via script)
├── models/                    # Serialized models and scalers (.pkl)
├── notebooks/                 
│   └── churn_analysis.ipynb   # Comprehensive EDA & ML pipeline
├── train_models.py            # Headless ML pipeline script
├── setup_data.py              # Script to fetch dataset
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   cd Customer-Churn-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Fetch the Dataset:**
   ```bash
   python setup_data.py
   ```

4. **Train the Models:**
   ```bash
   python train_models.py
   ```
   *(Alternatively, run the Jupyter Notebook `notebooks/churn_analysis.ipynb`)*

5. **Run the Dashboard:**
   ```bash
   streamlit run app/streamlit_dashboard.py
   ```

---

## 📈 Business Insights Highlights

- **Contract Type:** Month-to-month contracts exhibit the highest churn rates.
- **Internet Service:** Fiber optic users churn more frequently than DSL users, indicating potential pricing or quality issues.
- **Support Services:** Customers without Tech Support or Online Security are highly susceptible to leaving.
- **Lifetime Value:** Long-term customers with high LTV demonstrate significantly lower churn rates.

## 🔮 Future Improvements

- SHAP/LIME integration for deeper Model Explainability.
- Customer Segmentation using K-Means clustering.
- Integration with an SQLite/PostgreSQL database for real-time customer data streaming.
- Export prediction reports as PDF/CSV.

---
*Built for Data Science Executives and Analysts.*
