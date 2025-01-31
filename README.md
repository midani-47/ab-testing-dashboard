# A/B Testing Analysis Dashboard

A comprehensive dashboard for analyzing A/B test results with statistical rigor. This tool helps data scientists and analysts make data-driven decisions through statistical analysis and visualization of A/B test results.

## Features

- Statistical significance testing with configurable significance level (α)
- Effect size calculation (Cohen's d)
- Interactive visualizations including:
  - Distribution comparison
  - Box plots
  - Time series analysis
- Flexible metric selection (conversion rates, revenue)
- Real-time calculation of key statistics
- Interactive dashboard built with Streamlit
- Sample data generation for demonstration

## Installation

```bash
# Clone the repository
git clone https://github.com/midani-47/ab-testing-dashboard.git
cd ab-testing-dashboard

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

To run the dashboard:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

## Features Explained

### 1. Statistical Analysis
- **Significance Testing**: Uses Student's t-test to compare control and treatment groups
- **Effect Size**: Calculates Cohen's d to measure the magnitude of the difference
- **Configurable α**: Adjust significance level based on your needs

### 2. Visualizations
- **Distribution Plots**: Compare the distribution of metrics between groups
- **Box Plots**: Visualize key statistics and identify outliers
- **Time Series Analysis**: Track metrics over time for both groups

### 3. Metrics
- **Conversion Rate Analysis**: Compare conversion rates between groups
- **Revenue Analysis**: Analyze revenue differences
- **Sample Size Display**: Track the number of users in each group

## Sample Data

The dashboard includes a sample data generator that creates realistic A/B test data with:
- User IDs
- Group assignment (control/treatment)
- Conversion events
- Revenue data
- Timestamps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualization powered by [Plotly](https://plotly.com/)
- Statistical analysis using [SciPy](https://scipy.org/)