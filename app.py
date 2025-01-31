import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import io

# Page config
st.set_page_config(page_title='A/B Testing Dashboard', layout='wide')

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
This dashboard helps analyze A/B test results by providing:
- Statistical significance testing
- Effect size analysis 
- Visual comparisons between control and treatment groups
""")

# Sidebar controls
st.sidebar.header('Test Configuration')

# Data input method selection
data_input_method = st.sidebar.radio(
    "Choose data input method",
    ["Upload CSV", "Use Sample Data"]
)

# File upload or sample data generation
if data_input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file", 
        type=['csv'],
        help="File should contain columns: user_id, group (control/treatment), conversion (0/1), revenue (numeric), date"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            required_columns = ['user_id', 'group', 'conversion', 'revenue', 'date']
            if not all(col in data.columns for col in required_columns):
                st.error("CSV file must contain these columns: user_id, group, conversion, revenue, date")
                st.stop()
            data['date'] = pd.to_datetime(data['date'])
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.stop()
    else:
        st.warning("Please upload a CSV file or switch to sample data")
        st.stop()
else:
    # Sample data generation with configurable parameters
    st.sidebar.subheader("Sample Data Parameters")
    n_samples = st.sidebar.slider("Number of samples per group", 100, 5000, 1000)
    control_conv_rate = st.sidebar.slider("Control conversion rate", 0.01, 0.30, 0.10)
    treatment_conv_rate = st.sidebar.slider("Treatment conversion rate", 0.01, 0.30, 0.12)
    control_revenue_mean = st.sidebar.slider("Control revenue mean", 10.0, 100.0, 50.0)
    treatment_revenue_mean = st.sidebar.slider("Treatment revenue mean", 10.0, 100.0, 55.0)
    
    # Generate sample data with user parameters
    np.random.seed(None)  # Remove fixed seed
    
    # Control group data
    control_conversion = np.random.binomial(1, control_conv_rate, n_samples)
    control_revenue = np.random.exponential(control_revenue_mean, n_samples) * control_conversion
    
    # Treatment group data
    treatment_conversion = np.random.binomial(1, treatment_conv_rate, n_samples)
    treatment_revenue = np.random.exponential(treatment_revenue_mean, n_samples) * treatment_conversion
    
    # Create DataFrame
    data = pd.DataFrame({
        'user_id': range(2 * n_samples),
        'group': ['control'] * n_samples + ['treatment'] * n_samples,
        'conversion': np.concatenate([control_conversion, treatment_conversion]),
        'revenue': np.concatenate([control_revenue, treatment_revenue]),
        'date': pd.date_range(start='2025-01-01', periods=2*n_samples, freq='h')
    })

# Test configuration
alpha = st.sidebar.slider('Significance Level (α)', 0.01, 0.10, 0.05, 0.01)
metric_type = st.sidebar.selectbox(
    'Select Metric',
    ['conversion', 'revenue'],
    help="Conversion: Binary outcome (0/1) | Revenue: Continuous value in dollars"
)

# Display sample data
st.header('Data Preview')
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
    st.metric('P-value', f"{results['p_value']:.4f}",
              help="If p-value < α, the difference is statistically significant")
    
with col2:
    st.metric('Significant?', 'Yes' if results['significant'] else 'No',
              help=f"Based on significance level α={alpha}")
    
with col3:
    st.metric('Effect Size', f"{results['effect_size']:.4f}",
              help="Cohen's d: <0.2 small, 0.2-0.5 medium, >0.8 large")

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
relative_improvement = ((results['treatment_mean'] - results['control_mean']) / results['control_mean'] * 100)

st.markdown(f"""
### Key Findings:
- **Effect Size**: {abs(results['effect_size']):.3f} ({'favoring treatment' if results['effect_size'] > 0 else 'favoring control'})
  - {'Small' if abs(results['effect_size']) < 0.2 else 'Medium' if abs(results['effect_size']) < 0.8 else 'Large'} effect size
- **Relative Improvement**: {relative_improvement:.1f}% {('increase' if relative_improvement > 0 else 'decrease')} in {metric_type}
- **Statistical Significance**: The difference is{' ' if results['significant'] else ' not '} statistically significant at α={alpha}
  - Control group mean: {results['control_mean']:.4f}
  - Treatment group mean: {results['treatment_mean']:.4f}
""")

# Add download button for the data
st.header('Download Data')
csv = data.to_csv(index=False)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="ab_test_data.csv",
    mime="text/csv"
)