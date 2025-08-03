"""
Project Configuration Module

This module contains all project-wide configuration settings, paths, and constants.
Uses pathlib for cross-platform compatibility and relative path management.

Constants:
    ENTERPRISE_COLORS: Standard color scheme for ENT1/ENT2 visualizations
    BASE_PATHS: Core directory paths for data, output, and configuration
    FILE_SETTINGS: Default file formats and naming conventions

Example:
    from config.config import ENTERPRISE_COLORS, DRAFT_DIR
    
    # Use enterprise colors in plots
    color = ENTERPRISE_COLORS['ENT1']
    
    # Save to draft directory
    output_path = DRAFT_DIR / 'my_report.pdf'
"""

from pathlib import Path
from datetime import datetime
import os

# =============================================================================
# BASE PATH CONFIGURATION
# =============================================================================

# Get project root directory (parent of config folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directory paths
BASE_PATHS = {
    'DATA_ROOT': PROJECT_ROOT / '00_data',
    'NOTEBOOKS_ROOT': PROJECT_ROOT / '01_notebooks', 
    'MODULES_ROOT': PROJECT_ROOT / '02_modules',
    'OUTPUT_ROOT': PROJECT_ROOT / '03_output',
    'CONFIG_ROOT': PROJECT_ROOT / '04_config'
}

# Data directories
RAW_DATA_PATH = BASE_PATHS['DATA_ROOT'] / 'raw'
CLEANED_DATA_PATH = BASE_PATHS['DATA_ROOT'] / 'cleaned'

# Output directories
DRAFT_DIR = BASE_PATHS['OUTPUT_ROOT'] / 'draft'
FINAL_DIR = BASE_PATHS['OUTPUT_ROOT'] / 'final'

# Create timestamped subdirectories for outputs
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TIMESTAMPED_DRAFT = DRAFT_DIR / TIMESTAMP
TIMESTAMPED_FINAL = FINAL_DIR / TIMESTAMP

# =============================================================================
# VISUAL STYLING CONFIGURATION
# =============================================================================

# Enterprise color scheme for consistent visualizations
ENTERPRISE_COLORS = {
    'ENT1': '#1f77b4',      # Enterprise1 Blue
    'ENT2': '#ff7f0e',      # Enterprise2 Orange
    'NEUTRAL': '#2ca02c',   # Green for neutral/combined data
    'WARNING': '#d62728',   # Red for warnings/issues
    'INFO': '#9467bd',      # Purple for informational
    'SECONDARY': '#8c564b', # Brown for secondary data
    'LIGHT_GRAY': '#7f7f7f',
    'DARK_GRAY': '#17becf'
}

# Additional color palettes
SEQUENTIAL_COLORS = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', 
                    '#6baed6', '#4292c6', '#2171b5', '#084594']

DIVERGING_COLORS = ['#d73027', '#f46d43', '#fdae61', '#fee08b',
                   '#e6f598', '#abdda4', '#66c2a5', '#3288bd']

# Plot styling defaults
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_FONT_SIZE = 12

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

# File naming patterns
NAMING_CONVENTIONS = {
    'FIGURE_PREFIX': 'fig_',
    'TABLE_PREFIX': 'table_',
    'REPORT_PREFIX': 'report_',
    'DATA_PREFIX': 'data_'
}

# File extensions
FILE_EXTENSIONS = {
    'FIGURE_FORMATS': ['.png', '.pdf', '.svg'],
    'TABLE_FORMATS': ['.xlsx', '.csv'],
    'REPORT_FORMATS': ['.pdf', '.html']
}

# Default file settings
DEFAULT_SETTINGS = {
    'FIGURE_FORMAT': 'png',
    'TABLE_FORMAT': 'xlsx', 
    'REPORT_FORMAT': 'pdf',
    'EXCEL_ENGINE': 'openpyxl'
}

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Column mappings and standardizations
STANDARD_DATE_COLUMNS = [
    'origination_date', 'maturity_date', 'first_payment_date',
    'reporting_period', 'modification_date'
]

STANDARD_ID_COLUMNS = [
    'loan_sequence_number', 'bsa_id', 'property_zip_code',
    'borrower_ssn', 'servicer_id'
]

# Data validation rules
VALIDATION_RULES = {
    'DATE_RANGE_YEARS': 50,  # Maximum reasonable date range
    'MIN_LOAN_AMOUNT': 1000,  # Minimum loan amount
    'MAX_LOAN_AMOUNT': 50000000,  # Maximum loan amount
    'VALID_STATE_CODES': 51,  # Number of valid state codes (including DC)
}

# =============================================================================
# REPORTING CONFIGURATION
# =============================================================================

# Report metadata
REPORT_METADATA = {
    'AGENCY': 'Federal Housing Finance Agency',
    'DIVISION': 'Division of Housing Mission and Goals',
    'CONTACT': 'Nathan, Senior Data Scientist',
    'CLASSIFICATION': 'For Official Use Only'
}

# Excel formatting settings
EXCEL_FORMATTING = {
    'HEADER_COLOR': '#4472C4',
    'HEADER_FONT_COLOR': '#FFFFFF',
    'ZEBRA_COLOR': '#F2F2F2',
    'CURRENCY_FORMAT': '$#,##0',
    'PERCENTAGE_FORMAT': '0.00%',
    'DATE_FORMAT': 'MM/DD/YYYY'
}

# PDF report settings
PDF_SETTINGS = {
    'PAGE_SIZE': 'letter',
    'MARGINS': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
    'FONT_FAMILY': 'Arial',
    'TITLE_FONT_SIZE': 16,
    'BODY_FONT_SIZE': 11
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """
    Create all necessary directories if they don't exist.
    
    Returns:
        dict: Dictionary of created directory paths
    """
    directories_created = {}
    
    for name, path in BASE_PATHS.items():
        path.mkdir(parents=True, exist_ok=True)
        directories_created[name] = path
    
    # Create subdirectories
    for subdir in [RAW_DATA_PATH, CLEANED_DATA_PATH, DRAFT_DIR, FINAL_DIR]:
        subdir.mkdir(parents=True, exist_ok=True)
        directories_created[subdir.name] = subdir
    
    return directories_created

def get_timestamped_path(base_dir, create=True):
    """
    Generate a timestamped subdirectory path.
    
    Args:
        base_dir (Path): Base directory for timestamped folder
        create (bool): Whether to create the directory
        
    Returns:
        Path: Timestamped directory path
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_path = base_dir / timestamp
    
    if create:
        timestamped_path.mkdir(parents=True, exist_ok=True)
    
    return timestamped_path

def get_project_info():
    """
    Return project information dictionary.
    
    Returns:
        dict: Project metadata and configuration summary
    """
    return {
        'project_root': str(PROJECT_ROOT),
        'timestamp': TIMESTAMP,
        'enterprise_colors': ENTERPRISE_COLORS,
        'report_metadata': REPORT_METADATA,
        'data_paths': {
            'raw': str(RAW_DATA_PATH),
            'cleaned': str(CLEANED_DATA_PATH)
        },
        'output_paths': {
            'draft': str(DRAFT_DIR),
            'final': str(FINAL_DIR)
        }
    }

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Create directories and display project info when run directly
    print("Automated Reporting Project Configuration")
    print("=" * 50)
    
    # Ensure all directories exist
    created_dirs = ensure_directories()
    print(f"✓ Created {len(created_dirs)} directories")
    
    # Display project information
    info = get_project_info()
    print(f"✓ Project root: {info['project_root']}")
    print(f"✓ Timestamp: {info['timestamp']}")
    print(f"✓ Enterprise colors: {len(info['enterprise_colors'])} defined")
    
    print("\nConfiguration loaded successfully!")