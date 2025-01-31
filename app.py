import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime

def set_app_config():
    st.set_page_config(
        page_title='A/B Testing Dashboard',
        layout='wide',
        page_icon='📊'
    )

def calculate_z_test(success1, n1, success2, n2):
    """Manual implementation of proportion z-test with error handling"""
    if n1 == 0 or n2 == 0:
        return 0, 1.0  # Returning no effect if either sample is empty
    p1 = success1 / n1
    p2 = success2 / n2
    p_pooled = (success1 + success2) / (n1 + n2)
    # Handling the case where p_pooled is 0 or 1
    if p_pooled == 0 or p_pooled == 1:
        return 0, 1.0
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se if se != 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def calculate_power(effect_size, n1, n2, alpha=0.05):
    """Calculate statistical power"""
    return stats.norm.cdf(
        effect_size * np.sqrt(n1 * n2 / (n1 + n2)) - 
        stats.norm.ppf(1 - alpha/2)
    )

def calculate_significance(control_data, treatment_data, alpha=0.05, metric_type='conversion'):
    """Calculate statistical significance with appropriate tests"""
    # Handle empty data case
    if len(control_data) == 0 or len(treatment_data) == 0:
        return {
            'p_value': 1.0,
            'significant': False,
            'effect_size': 0,
            'control_mean': 0,
            'treatment_mean': 0,
            'relative_improvement': 0,
            'confidence_interval': (0, 0),
            'power': 0
        }

    control_mean = np.mean(control_data)
    treatment_mean = np.mean(treatment_data)
    
    if metric_type == 'conversion':
        # Manual z-test for proportions
        success1 = sum(treatment_data)
        success2 = sum(control_data)
        n1, n2 = len(treatment_data), len(control_data)
        z_stat, p_value = calculate_z_test(success1, n1, success2, n2)
        
        # Calculate Cohen's h for binary data
        h = 2 * (np.arcsin(np.sqrt(max(0, min(1, treatment_mean)))) - np.arcsin(np.sqrt(max(0, min(1, control_mean)))))
        effect_size = h
    else:
        # Use t-test for continuous data
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        
        # Calculate Cohen's d
        pooled_std = np.sqrt(
            (np.std(control_data, ddof=1)**2 + np.std(treatment_data, ddof=1)**2) / 2
        )
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std != 0 else 0

    # Calculate confidence intervals with error handling
    try:
        if metric_type == 'conversion':
            # Ensure treatment_mean is between 0 and 1
            adj_treatment_mean = max(0, min(1, treatment_mean))
            if adj_treatment_mean == 0 or adj_treatment_mean == 1 or len(treatment_data) == 0:
                ci = (adj_treatment_mean, adj_treatment_mean)
            else:
                ci = stats.norm.interval(
                    1 - alpha,
                    adj_treatment_mean,
                    np.sqrt(adj_treatment_mean * (1 - adj_treatment_mean) / len(treatment_data))
                )
        else:
            if len(treatment_data) <= 1:
                ci = (treatment_mean, treatment_mean)
            else:
                ci = stats.t.interval(
                    1 - alpha,
                    len(treatment_data) - 1,
                    treatment_mean,
                    stats.sem(treatment_data)
                )
    except (ValueError, ZeroDivisionError):
        ci = (treatment_mean, treatment_mean)

    return {
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': effect_size,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'relative_improvement': (treatment_mean - control_mean) / control_mean if control_mean != 0 else 0,
        'confidence_interval': ci,
        'power': calculate_power(effect_size, len(control_data), len(treatment_data), alpha)
    }

def plot_trend(data, metric_type):
    """Plot trend over time with proper handling of rolling average"""
    # First, ensure the data is properly sorted
    trend_data = data.copy()
    trend_data['date'] = pd.to_datetime(trend_data['date'])
    trend_data = trend_data.sort_values('date')
    
    # Calculate daily averages first
    daily_data = (trend_data.groupby(['date', 'group'])[metric_type]
                 .mean()
                 .reset_index())
    
    # Calculate 7-day rolling average
    rolling_data = []
    for group in daily_data['group'].unique():
        group_data = daily_data[daily_data['group'] == group].copy()
        group_data[f'{metric_type}_rolling'] = (
            group_data[metric_type]
            .rolling(window=7, min_periods=1)
            .mean()
        )
        rolling_data.append(group_data)
    
    rolling_data = pd.concat(rolling_data)
    
    # Create the plot
    fig = px.line(
        rolling_data,
        x='date',
        y=f'{metric_type}_rolling',
        color='group',
        title=f'{metric_type.title()} Trend Over Time (7-day Moving Average)',
        labels={f'{metric_type}_rolling': metric_type}
    )
    return fig

@st.cache_data
def load_data(uploaded_file):
    """Load and validate uploaded data"""
    try:
        data = pd.read_csv(uploaded_file)
        required_columns = {'user_id', 'group', 'conversion', 'revenue', 'date'}
        if not required_columns.issubset(set(data.columns)):
            st.error(f"CSV must contain {', '.join(required_columns)}")
            return None
        
        data['date'] = pd.to_datetime(data['date'])
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_sample_data(n_samples, control_params, treatment_params):
    """Generate realistic sample data"""
    np.random.seed(int(datetime.now().timestamp()))
    
    data = []
    for group, params in [('control', control_params), ('treatment', treatment_params)]:
        conversions = np.random.binomial(1, params['conv_rate'], n_samples)
        revenue = np.where(
            conversions == 1,
            np.random.gamma(shape=2, scale=params['revenue_mean']/2, size=n_samples),
            0
        )
        
        data.append(pd.DataFrame({
            'user_id': [f"{group}_{i}" for i in range(n_samples)],
            'group': group,
            'conversion': conversions,
            'revenue': np.round(revenue, 2),
            'date': pd.date_range(end=datetime.today(), periods=n_samples, freq='min')
        }))
    
    return pd.concat(data, ignore_index=True)

def main():
    set_app_config()
    
    st.title('A/B Testing Analysis Dashboard')
    st.markdown("""
    This dashboard provides comprehensive A/B test analysis with:
    - **Statistical Significance Testing** (Z-test for conversions, T-test for revenue)
    - **Effect Size Analysis** (Cohen's h/d)
    - **Confidence Intervals**
    - **Statistical Power Analysis**
    - **Visualizations** tailored to metric type
    - **Business Impact Estimates**
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header('Test Configuration')
        data_input_method = st.radio(
            "Data Input Method",
            ["Upload CSV", "Use Sample Data"],
            help="Choose between uploading your data or exploring with generated sample data"
        )
        
        if data_input_method == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=['csv'],
                help="Required columns: user_id, group, conversion, revenue, date"
            )
            if uploaded_file:
                data = load_data(uploaded_file)
                if data is None:
                    st.stop()
            else:
                st.info("Please upload a CSV file to continue")
                st.stop()
        else:
            st.subheader("Sample Data Parameters")
            n_samples = st.slider("Samples per group", 500, 5000, 1500, 500)
            with st.expander("Control Group Parameters"):
                control_conv = st.slider("Conversion Rate", 0.01, 0.30, 0.12, key='control_conv')
                control_rev = st.slider("Revenue Mean", 10.0, 100.0, 60.0, key='control_rev')
            
            with st.expander("Treatment Group Parameters"):
                treat_conv = st.slider("Conversion Rate", 0.01, 0.30, 0.15, key='treat_conv')
                treat_rev = st.slider("Revenue Mean", 10.0, 100.0, 65.0, key='treat_rev')
            
            data = generate_sample_data(
                n_samples,
                {'conv_rate': control_conv, 'revenue_mean': control_rev},
                {'conv_rate': treat_conv, 'revenue_mean': treat_rev}
            )
        
        alpha = st.slider('Significance Level (α)', 0.01, 0.10, 0.05, 0.01)
        metric_type = st.selectbox(
            'Analysis Metric',
            ['conversion', 'revenue'],
            help="Select the primary metric for analysis"
        )

    # Data preview
    st.header('Data Overview')
    with st.expander("Preview Raw Data"):
        st.dataframe(data.head(), use_container_width=True)
    
    with st.expander("Data Summary"):
        st.write(f"Total Records: {len(data):,}")
        st.write(f"Date Range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
        st.dataframe(
            data.groupby('group').agg({
                'conversion': ['count', 'mean', 'sum'],
                'revenue': ['mean', 'sum', 'std']
            }).style.format(precision=3),
            use_container_width=True
        )

    # Statistical analysis
    st.header('Statistical Results')
    control_data = data[data['group'] == 'control'][metric_type]
    treatment_data = data[data['group'] == 'treatment'][metric_type]
    
    results = calculate_significance(control_data, treatment_data, alpha, metric_type)
    
    # Display metrics
    cols = st.columns(4)
    metric_config = {
        'p_value': {'label': 'P-value', 'format': '{:.4f}'},
        'significant': {'label': 'Significant?', 'format': lambda x: 'Yes' if x else 'No'},
        'effect_size': {'label': 'Effect Size', 'format': '{:.3f}',
                       'help': "Cohen's h (conversion) or d (revenue)"},
        'power': {'label': 'Statistical Power', 'format': '{:.1%}',
                 'help': 'Probability of detecting the observed effect size'}
    }

    for (k, config), col in zip(metric_config.items(), cols):
        with col:
            value = results[k]
            formatted = config['format'].format(value) if isinstance(config['format'], str) else config['format'](value)
            st.metric(
                label=config['label'],
                value=formatted,
                help=config.get('help')
            )

    # Confidence interval
    st.metric(
        label="95% Confidence Interval",
        value=f"({results['confidence_interval'][0]:.3f}, {results['confidence_interval'][1]:.3f})",
        help="95% confidence interval for the treatment group metric"
    )

    # Visualizations
    st.header('Data Visualization')
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if metric_type == 'conversion':
            fig = px.bar(
                data.groupby('group', as_index=False)['conversion'].mean(),
                x='group',
                y='conversion',
                title='Conversion Rate Comparison',
                color='group',
                text_auto='.1%'
            )
        else:
            fig = px.box(
                data[data['revenue'] > 0],
                x='group',
                y='revenue',
                title='Revenue Distribution (Converted Users)',
                color='group'
            )
        st.plotly_chart(fig, use_container_width=True)

    with viz_col2:
        trend_fig = plot_trend(data, metric_type)
        st.plotly_chart(trend_fig, use_container_width=True)

    # Business impact analysis
    if results['significant'] and metric_type == 'conversion':
        st.header('Business Impact Estimation')
        col1, col2, col3 = st.columns(3)
        
        avg_visitors = st.number_input("Estimated Monthly Visitors", 10000, 1000000, 50000)
        avg_order_value = data[data['conversion'] == 1]['revenue'].mean()
        
        with col1:
            st.metric(
                "Estimated Monthly Conversions",
                f"{(avg_visitors * results['treatment_mean']):,.0f}",
                delta=f"{(results['relative_improvement'] * avg_visitors * results['control_mean']):+.0f} vs Control"
            )
        
        with col2:
            if avg_order_value > 0:
                st.metric(
                    "Estimated Monthly Revenue Impact",
                    f"${avg_visitors * results['treatment_mean'] * avg_order_value:,.0f}",
                    delta=f"${results['relative_improvement'] * avg_visitors * results['control_mean'] * avg_order_value:+.0f}"
                )
        
        with col3:
            required_n = int(16 * (1.96 + 0.84)**2 / (results['effect_size']**2))
            st.metric(
                "Required Sample Size",
                f"{required_n:,}",
                help="Sample size needed per group for 80% power"
            )

    # Data export
    st.download_button(
        label="Download Processed Data",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="ab_test_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()