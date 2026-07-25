import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Model Loading ---
def load_artifacts(models_dir='models'):
    """Loads the ML model and preprocessors."""
    best_model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
    return best_model, scaler, label_encoders, feature_cols

# --- Plotly Theme ---
def apply_custom_theme(fig):
    """Applies a premium dark theme to Plotly figures."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(size=20, color='#3b82f6', family='Inter'),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        legend=dict(bgcolor='rgba(11, 15, 25, 0.7)', bordercolor='rgba(255,255,255,0.1)'),
        hovermode='x unified'
    )
    return fig

# --- Insight Generator ---
def get_insight(section):
    """Returns business insights based on the analysis section."""
    insights = {
        "demographics": "Customers with Month-to-Month contracts have the highest churn rate. Senior citizens represent a smaller portion of the customer base but churn at a higher relative percentage.",
        "revenue": "High monthly charges correlate strongly with churn. Customers paying over $70/month require targeted retention strategies. Conversely, long-term customers with high LTV (Lifetime Value) show significantly lower churn.",
        "services": "Fiber-optic internet users have a noticeably higher churn percentage compared to DSL, potentially indicating service quality or pricing issues. Customers without additional services like Technical Support or Online Security are highly susceptible to leaving.",
        "predictions": "The Machine Learning model highlights 'Contract Type', 'Tenure', and 'Total Charges' as the most critical factors influencing churn."
    }
    return f'<div class="insight-box">💡 <b>Business Insight:</b> {insights.get(section, "Analyze the chart above to identify key trends.")}</div>'
