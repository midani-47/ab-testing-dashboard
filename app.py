import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# Page config
st.set_page_config(page_title='A/B Testing Dashboard', layout='wide')

def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Control group data
    control_conversion = np.random.binomial(1, 0.10, n_samples)
    control_revenue = np.random.exponential(50, n_samples) * control_conversion
    
    # Treatment group data (with slight improvement)
    treatment_conversion = np.random.binomial(1, 0.12, n_samples)
    treatment_revenue = np.random.exponential(55, n_samples) * treatment_conversion
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': range(2 * n_samples),
        'group': ['control'] * n_samples + ['treatment'] * n_samples,
        'conversion': np.concatenate([control_conversion, treatment_conversion]),
        'revenue': np.concatenate([control_revenue, treatment_revenue]),
        'date': pd.date_range(start='2024-01-01', periods=2*n_samples, freq='H')
    })
    
    return df

def calculate_significance(control_data, treatment_data, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(control_data), len(treatment_data)
    pooled_std = np.sqrt(((n1-1)*np.std(control_data, ddof=1)**2 + 
                         (n2-1)*np.std(treatment_data, ddof=1)**2) / 
                        (n1+n2-2))
    cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
    
    return {
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohens_d,
        'control_mean': np.mean(control_data),
        'treatment_mean': np.mean(treatment_data)
    }

# Title and description
st.title('A/B Testing Analysis Dashboard')
st.markdown("""
This dashboard provides comprehensive analysis of A/B test results including 
statistical significance testing, effect size analysis, and visualizations.
""")

# Sidebar controls
st.sidebar.header('Test Configuration')
alpha = st.sidebar.slider('Significance Level (α)', 0.01, 0.10, 0.05, 0.01)
metric_type = st.sidebar.selectbox(
    'Select Metric',
    ['conversion', 'revenue']
)

# Load data
data = generate_sample_data()

# Display sample data
st.header('Sample Data Preview')
st.dataframe(data.head())

# Calculate and display metrics
metrics = data.groupby('group').agg({
    metric_type: ['count', 'mean', 'std']
}).round(4)
metrics.columns = ['Sample Size', 'Mean', 'Std Dev']

st.header('Metrics Summary')
st.dataframe(metrics)

# Statistical Analysis
st.header('Statistical Analysis')

control_data = data[data['group'] == 'control'][metric_type]
treatment_data = data[data['group'] == 'treatment'][metric_type]

# Calculate results
results = calculate_significance(control_data, treatment_data, alpha)

# Display results in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('P-value', f"{results['p_value']:.4f}")
    
with col2:
    st.metric('Significant?', 'Yes' if results['significant'] else 'No')
    
with col3:
    st.metric('Effect Size', f"{results['effect_size']:.4f}")

# Visualizations
st.header('Visualizations')

# Distribution plot
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=control_data,
    name='Control',
    opacity=0.75
))
fig.add_trace(go.Histogram(
    x=treatment_data,
    name='Treatment',
    opacity=0.75
))
fig.update_layout(
    title=f'{metric_type.capitalize()} Distribution by Group',
    xaxis_title=metric_type.capitalize(),
    yaxis_title='Count',
    barmode='overlay'
)

st.plotly_chart(fig, use_container_width=True)

# Box plot
fig = px.box(data, x='group', y=metric_type,
             title=f'{metric_type.capitalize()} Box Plot by Group')
st.plotly_chart(fig, use_container_width=True)

# Time series analysis
daily_metrics = data.groupby(['date', 'group'])[metric_type].mean().reset_index()

fig = px.line(daily_metrics, x='date', y=metric_type, color='group',
              title=f'{metric_type.capitalize()} Over Time by Group')
st.plotly_chart(fig, use_container_width=True)

# Additional insights
st.header('Additional Insights')
st.markdown(f"""
- The treatment group shows a {results['effect_size']:.2%} effect size compared to the control group
- Mean {metric_type} for control group: {results['control_mean']:.4f}
- Mean {metric_type} for treatment group: {results['treatment_mean']:.4f}
- The difference is{' ' if results['significant'] else ' not '} statistically significant at α={alpha}
""")