# Automated Reporting Project Template

## Project Overview
This template provides a standardized, reusable folder structure for automated regulatory reporting and analytics projects. Designed for a regulatory agency, it supports end-to-end data processing workflows from raw loan-level data to professional PDF and Excel reports.

## 🗂️ Folder Structure

```
automated_report_project/
├── 00_data/                      # Raw and intermediate input data
│   ├── raw/                      # Original data files (CSV, Excel, SQL dumps)
│   │   ├── .gitkeep
│   │   └── README.md
│   └── cleaned/                  # Processed and validated data
│       ├── .gitkeep
│       └── README.md
├── 01_notebooks/                 # Jupyter notebooks by report section
│   ├── 1_data_overview.ipynb     # Data loading and initial exploration
│   ├── 2_trend_analysis.ipynb    # Analytical deep-dives and visualizations
│   ├── 3_final_report.ipynb      # Report compilation and export
│   └── notebook_template.ipynb   # Starter template for new sections
├── 02_modules/                   # Reusable Python modules
│   ├── __init__.py
│   ├── report_utils.py           # Core reporting utilities
│   ├── cleaning.py               # Data cleaning functions
│   ├── validation.py             # Data validation and quality checks
│   └── visuals.py                # Standardized plotting functions
├── 03_output/                    # Generated reports and exports
│   ├── draft/                    # Work-in-progress outputs
│   │   └── .gitkeep
│   └── final/                    # Publication-ready reports
│       └── .gitkeep
├── 04_config/                    # Configuration and settings
│   ├── __init__.py
│   └── config.py                 # Project configuration
├── tests/                        # Unit tests and validation scripts
│   ├── __init__.py
│   ├── test_cleaning.py
│   ├── test_validation.py
│   └── sample_data/
│       └── .gitkeep
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
└── setup.py                      # Package installation script
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or download this template
git clone <repository-url>
cd automated_report_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
1. Update `04_config/config.py` with your project-specific paths and settings
2. Place raw data files in `00_data/raw/`
3. Review and customize the Jupyter notebooks in `01_notebooks/`

### 3. Running the Project
```python
# In Jupyter or Python script
from modules.report_utils import generate_all_outputs
from config.config import *

# Generate all reports
generate_all_outputs()
```

## 📚 Module Documentation

### report_utils.py
Core utilities for figure/table saving, export formatting, and output management.

**Key Functions:**
- `save_figure()`: Save plots with timestamped directories
- `save_table()`: Export formatted tables to Excel
- `print_to_pdf()`: Generate PDF reports
- `generate_all_outputs()`: Rebuild all project outputs

### cleaning.py
Data preprocessing and standardization functions.

**Key Functions:**
- `promote_headers()`: Standardize column headers
- `clean_dates()`: Parse and validate date fields
- `standardize_ids()`: Format loan numbers and identifiers

### validation.py
Data quality checks and validation rules.

**Key Functions:**
- `validate_date_ranges()`: Check for date inconsistencies
- `flag_invalid_records()`: Identify problematic data
- `generate_quality_report()`: Summarize data quality metrics

### visuals.py
Standardized plotting functions with consistent styling.

**Key Functions:**
- `plot_time_series()`: Time-based trend analysis
- `plot_choropleth()`: Geographic visualizations
- `plot_horizontal_bars()`: Comparative bar charts

## 🎨 Visual Standards

### Color Scheme
- **Fannie Mae (FNM)**: `#1f77b4` (Blue)
- **Freddie Mac (FRE)**: `#ff7f0e` (Orange)
- **Additional colors**: Defined in `ENTERPRISE_COLORS` dictionary

### File Naming Conventions
- Figures: `fig_[number][letter]_[description].png`
- Tables: `table_[number]_[description].xlsx`
- Reports: `report_[YYYYMMDD]_[description].pdf`

## 🔧 Dependencies

**Core Libraries:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- plotly >= 5.0.0
- openpyxl >= 3.0.7
- jupyter >= 1.0.0

**Additional Tools:**
- pathlib (built-in)
- datetime (built-in)
- os (built-in)

## 📖 Usage Examples

### Basic Data Processing
```python
from modules.cleaning import promote_headers, clean_dates
from modules.validation import validate_date_ranges
from config.config import RAW_DATA_PATH

# Load and clean data
df = pd.read_csv(RAW_DATA_PATH / "loan_data.csv")
df = promote_headers(df)
df = clean_dates(df, date_columns=['origination_date', 'maturity_date'])

# Validate data quality
validation_results = validate_date_ranges(df)
```

### Generate Visualizations
```python
from modules.visuals import plot_time_series
from modules.report_utils import save_figure

# Create time series plot
fig = plot_time_series(df, x_col='date', y_col='volume', 
                      title='Loan Volume Trends')

# Save with automatic timestamping
save_figure(fig, 'fig_1a_loan_volume_trends', print_mode='YES')
```
