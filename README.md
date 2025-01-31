# A/B Testing Analysis Dashboard

A comprehensive dashboard for analyzing A/B test results with statistical rigor. This tool helps data scientists and analysts make data-driven decisions through statistical analysis and visualization of A/B test results.

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

## Quick Start

### For macOS/Linux:
```bash
# Clone the repository
git clone https://github.com/midani-47/ab-testing-dashboard.git
cd ab-testing-dashboard

# Run the setup script
chmod +x setup.sh
./setup.sh
```

### For Windows:
```bash
# Clone the repository
git clone https://github.com/midani-47/ab-testing-dashboard.git
cd ab-testing-dashboard

# Run the setup script
setup.bat
```

After installation, run the dashboard:
```bash
# The setup script will create and activate a virtual environment
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

## Manual Installation (if setup script doesn't work)

```bash
# Clone the repository
git clone https://github.com/midani-47/ab-testing-dashboard.git
cd ab-testing-dashboard

# Create and activate virtual environment
python -m venv venv

# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure Python 3.8+ is installed:
   ```bash
   python --version
   ```

2. For macOS users, if you get compilation errors, install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```

3. For Windows users, make sure you have the latest pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. If streamlit command is not found, try using the full path:
   ```bash
   # Windows
   .\venv\Scripts\streamlit.exe run app.py
   # macOS/Linux
   ./venv/bin/streamlit run app.py
   ```

## Features

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