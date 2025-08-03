"""
Data Cleaning and Standardization Module

This module provides functions for cleaning and standardizing loan-level data
from Enterprise1 and Enterprise2, including header promotion, date parsing,
ID field standardization, and data type conversions.

Functions:
    promote_headers: Standardize and clean column headers
    clean_dates: Parse and validate date columns
    standardize_ids: Format loan numbers and identifier fields
    clean_numeric_fields: Handle numeric data with proper types
    remove_invalid_records: Filter out records with critical errors
    standardize_categorical: Clean categorical variables

Example:
    from modules.cleaning import promote_headers, clean_dates
    
    # Clean raw data
    df = promote_headers(df)
    df = clean_dates(df, ['origination_date', 'maturity_date'])
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import STANDARD_DATE_COLUMNS, STANDARD_ID_COLUMNS, VALIDATION_RULES

# =============================================================================
# HEADER AND COLUMN CLEANING
# =============================================================================

def promote_headers(df, header_row=0, strip_whitespace=True, lowercase=False):
    """
    Promote and standardize DataFrame column headers.
    
    Args:
        df (DataFrame): Input DataFrame
        header_row (int): Row index to use as headers
        strip_whitespace (bool): Remove leading/trailing whitespace
        lowercase (bool): Convert headers to lowercase
        
    Returns:
        DataFrame: DataFrame with cleaned headers
        
    Example:
        df = promote_headers(df, header_row=0, strip_whitespace=True)
    """
    df_clean = df.copy()
    
    # Set headers from specified row if not already headers
    if header_row > 0:
        df_clean.columns = df_clean.iloc[header_row]
        df_clean = df_clean.iloc[header_row + 1:].reset_index(drop=True)
    
    # Clean column names
    new_columns = []
    for col in df_clean.columns:
        clean_col = str(col)
        
        if strip_whitespace:
            clean_col = clean_col.strip()
        
        if lowercase:
            clean_col = clean_col.lower()
        
        # Replace common problematic characters
        clean_col = clean_col.replace(' ', '_')
        clean_col = clean_col.replace('-', '_')
        clean_col = clean_col.replace('/', '_')
        clean_col = clean_col.replace('(', '').replace(')', '')
        clean_col = re.sub(r'[^\w_]', '', clean_col)
        clean_col = re.sub(r'_+', '_', clean_col)
        clean_col = clean_col.strip('_')
        
        new_columns.append(clean_col)
    
    df_clean.columns = new_columns
    
    # Remove completely empty columns
    df_clean = df_clean.dropna(axis=1, how='all')
    
    print(f"✓ Headers promoted and cleaned: {len(df_clean.columns)} columns")
    return df_clean

def standardize_column_names(df, mapping_dict=None):
    """
    Apply standard column name mappings for consistency.
    
    Args:
        df (DataFrame): Input DataFrame
        mapping_dict (dict): Custom column name mappings
        
    Returns:
        DataFrame: DataFrame with standardized column names
        
    Example:
        mapping = {'loan_id': 'loan_sequence_number'}
        df = standardize_column_names(df, mapping)
    """
    if mapping_dict is None:
        # Default mappings for common variations
        mapping_dict = {
            'loan_id': 'loan_sequence_number',
            'origination_dt': 'origination_date',
            'maturity_dt': 'maturity_date',
            'first_payment_dt': 'first_payment_date',
            'zip': 'property_zip_code',
            'zip_code': 'property_zip_code',
            'state': 'property_state',
            'enterprise': 'enterprise_flag',
            'loan_amt': 'loan_amount',
            'loan_amount_000': 'loan_amount'
        }
    
    df_clean = df.copy()
    
    # Apply mappings
    columns_renamed = 0
    for old_name, new_name in mapping_dict.items():
        if old_name in df_clean.columns:
            df_clean = df_clean.rename(columns={old_name: new_name})
            columns_renamed += 1
    
    print(f"✓ Column names standardized: {columns_renamed} columns renamed")
    return df_clean

# =============================================================================
# DATE CLEANING AND PARSING
# =============================================================================

def clean_dates(df, date_columns=None, date_formats=None, handle_errors='coerce'):
    """
    Parse and clean date columns with multiple format handling.
    
    Args:
        df (DataFrame): Input DataFrame
        date_columns (list): List of date column names (default: STANDARD_DATE_COLUMNS)
        date_formats (list): List of date formats to try
        handle_errors (str): How to handle parsing errors ('coerce', 'raise', 'warn')
        
    Returns:
        DataFrame: DataFrame with parsed date columns
        
    Example:
        df = clean_dates(df, ['origination_date', 'maturity_date'])
    """
    if date_columns is None:
        date_columns = [col for col in STANDARD_DATE_COLUMNS if col in df.columns]
    
    if date_formats is None:
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%Y%m%d',
            '%m/%d/%y',
            '%Y-%m-%d %H:%M:%S'
        ]
    
    df_clean = df.copy()
    parsing_summary = {}
    
    for col in date_columns:
        if col not in df_clean.columns:
            continue
            
        original_count = len(df_clean[col].dropna())
        successful_parses = 0
        
        # Try each date format
        for date_format in date_formats:
            try:
                # Parse dates that haven't been parsed yet
                mask = df_clean[col].notna() & df_clean[col].astype(str).str.match(r'^\d')
                if mask.any():
                    parsed_dates = pd.to_datetime(
                        df_clean.loc[mask, col], 
                        format=date_format, 
                        errors='coerce'
                    )
                    
                    # Update successfully parsed dates
                    valid_mask = parsed_dates.notna()
                    df_clean.loc[mask & valid_mask, col] = parsed_dates[valid_mask]
                    successful_parses += valid_mask.sum()
                    
            except (ValueError, TypeError):
                continue
        
        # Final attempt with flexible parsing
        remaining_mask = df_clean[col].notna() & df_clean[col].astype(str).str.match(r'^\d')
        if remaining_mask.any():
            try:
                flexible_parsed = pd.to_datetime(
                    df_clean.loc[remaining_mask, col], 
                    errors='coerce',
                    infer_datetime_format=True
                )
                valid_flexible = flexible_parsed.notna()
                df_clean.loc[remaining_mask & valid_flexible, col] = flexible_parsed[valid_flexible]
                successful_parses += valid_flexible.sum()
                
            except Exception:
                pass
        
        # Handle parsing errors
        final_count = len(df_clean[col].dropna())
        failed_count = original_count - successful_parses
        
        parsing_summary[col] = {
            'original': original_count,
            'parsed': successful_parses,
            'failed': failed_count,
            'success_rate': successful_parses / original_count if original_count > 0 else 0
        }
        
        if handle_errors == 'warn' and failed_count > 0:
            print(f"⚠️  {col}: {failed_count} dates failed to parse")
        elif handle_errors == 'raise' and failed_count > 0:
            raise ValueError(f"Failed to parse {failed_count} dates in column {col}")
    
    # Print summary
    total_original = sum(s['original'] for s in parsing_summary.values())
    total_parsed = sum(s['parsed'] for s in parsing_summary.values())
    
    print(f"✓ Date parsing complete: {total_parsed}/{total_original} dates parsed "
          f"({total_parsed/total_original*100:.1f}% success rate)")
    
    return df_clean

def validate_date_ranges(df, date_columns=None, min_year=1970, max_year=None):
    """
    Validate that dates fall within reasonable ranges.
    
    Args:
        df (DataFrame): Input DataFrame
        date_columns (list): Date columns to validate
        min_year (int): Minimum acceptable year
        max_year (int): Maximum acceptable year (default: current year + 50)
        
    Returns:
        DataFrame: DataFrame with invalid dates set to NaT
        
    Example:
        df = validate_date_ranges(df, min_year=1990, max_year=2030)
    """
    if date_columns is None:
        date_columns = [col for col in STANDARD_DATE_COLUMNS if col in df.columns]
    
    if max_year is None:
        max_year = datetime.now().year + VALIDATION_RULES['DATE_RANGE_YEARS']
    
    df_clean = df.copy()
    validation_summary = {}
    
    for col in date_columns:
        if col not in df_clean.columns:
            continue
        
        original_count = df_clean[col].notna().sum()
        
        # Validate year ranges
        year_mask = (
            (df_clean[col].dt.year >= min_year) & 
            (df_clean[col].dt.year <= max_year)
        )
        
        # Set invalid dates to NaT
        df_clean.loc[~year_mask, col] = pd.NaT
        
        final_count = df_clean[col].notna().sum()
        invalid_count = original_count - final_count
        
        validation_summary[col] = {
            'original': original_count,
            'valid': final_count,
            'invalid': invalid_count
        }
    
    total_invalid = sum(s['invalid'] for s in validation_summary.values())
    if total_invalid > 0:
        print(f"⚠️  Date validation: {total_invalid} dates outside valid range removed")
    
    return df_clean

# =============================================================================
# ID FIELD STANDARDIZATION
# =============================================================================

def standardize_ids(df, id_columns=None, formats=None):
    """
    Standardize ID fields with proper formatting and data types.
    
    Args:
        df (DataFrame): Input DataFrame
        id_columns (list): List of ID column names
        formats (dict): Formatting rules for each ID type
        
    Returns:
        DataFrame: DataFrame with standardized ID fields
        
    Example:
        df = standardize_ids(df, ['loan_sequence_number', 'bsa_id'])
    """
    if id_columns is None:
        id_columns = [col for col in STANDARD_ID_COLUMNS if col in df.columns]
    
    if formats is None:
        formats = {
            'loan_sequence_number': {'type': 'string', 'pad': None},
            'bsa_id': {'type': 'string', 'pad': None},
            'property_zip_code': {'type': 'string', 'pad': 5},
            'borrower_ssn': {'type': 'string', 'pad': 9},
            'servicer_id': {'type': 'string', 'pad': None}
        }
    
    df_clean = df.copy()
    standardization_summary = {}
    
    for col in id_columns:
        if col not in df_clean.columns:
            continue
        
        original_count = len(df_clean[col].dropna())
        
        # Convert to string and clean
        df_clean[col] = df_clean[col].astype(str)
        
        # Remove common problematic characters
        df_clean[col] = df_clean[col].str.replace(r'[^\w]', '', regex=True)
        
        # Apply specific formatting if defined
        if col in formats:
            format_rules = formats[col]
            
            # Pad with zeros if specified
            if format_rules.get('pad'):
                pad_length = format_rules['pad']
                df_clean[col] = df_clean[col].str.zfill(pad_length)
            
            # Convert to appropriate type
            if format_rules.get('type') == 'int':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Replace 'nan' strings with actual NaN
        df_clean[col] = df_clean[col].replace('nan', np.nan)
        
        final_count = len(df_clean[col].dropna())
        standardization_summary[col] = {
            'original': original_count,
            'standardized': final_count
        }
    
    print(f"✓ ID standardization complete: {len(id_columns)} columns processed")
    return df_clean

# =============================================================================
# NUMERIC FIELD CLEANING
# =============================================================================

def clean_numeric_fields(df, numeric_columns=None, remove_commas=True, 
                        handle_negatives='keep', validation_rules=None):
    """
    Clean and standardize numeric fields.
    
    Args:
        df (DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names
        remove_commas (bool): Remove comma separators
        handle_negatives (str): How to handle negative values ('keep', 'abs', 'remove')
        validation_rules (dict): Min/max validation rules
        
    Returns:
        DataFrame: DataFrame with cleaned numeric fields
        
    Example:
        df = clean_numeric_fields(df, ['loan_amount', 'property_value'])
    """
    if numeric_columns is None:
        # Auto-detect likely numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add common loan data numeric fields
        potential_numeric = [
            'loan_amount', 'property_value', 'ltv_ratio', 'dti_ratio',
            'credit_score', 'interest_rate', 'loan_term'
        ]
        
        for col in potential_numeric:
            if col in df.columns and col not in numeric_columns:
                numeric_columns.append(col)
    
    if validation_rules is None:
        validation_rules = {
            'loan_amount': {'min': VALIDATION_RULES['MIN_LOAN_AMOUNT'], 
                          'max': VALIDATION_RULES['MAX_LOAN_AMOUNT']},
            'property_value': {'min': 1000, 'max': 50000000},
            'ltv_ratio': {'min': 0, 'max': 200},
            'dti_ratio': {'min': 0, 'max': 100},
            'credit_score': {'min': 300, 'max': 850},
            'interest_rate': {'min': 0, 'max': 30}
        }
    
    df_clean = df.copy()
    cleaning_summary = {}
    
    for col in numeric_columns:
        if col not in df_clean.columns:
            continue
        
        original_count = len(df_clean[col].dropna())
        
        # Convert to string first for cleaning
        col_str = df_clean[col].astype(str)
        
        # Remove commas and dollar signs
        if remove_commas:
            col_str = col_str.str.replace(',', '')
            col_str = col_str.str.replace('$', '')
        
        # Handle percentage signs
        percentage_mask = col_str.str.contains('%', na=False)
        col_str = col_str.str.replace('%', '')
        
        # Convert to numeric
        numeric_series = pd.to_numeric(col_str, errors='coerce')
        
        # Convert percentages to decimals
        numeric_series.loc[percentage_mask] = numeric_series.loc[percentage_mask] / 100
        
        # Handle negative values
        if handle_negatives == 'abs':
            numeric_series = numeric_series.abs()
        elif handle_negatives == 'remove':
            numeric_series = numeric_series.where(numeric_series >= 0)
        
        # Apply validation rules
        if col in validation_rules:
            rules = validation_rules[col]
            if 'min' in rules:
                numeric_series = numeric_series.where(numeric_series >= rules['min'])
            if 'max' in rules:
                numeric_series = numeric_series.where(numeric_series <= rules['max'])
        
        # Update DataFrame
        df_clean[col] = numeric_series
        
        final_count = len(df_clean[col].dropna())
        cleaning_summary[col] = {
            'original': original_count,
            'cleaned': final_count,
            'removed': original_count - final_count
        }
    
    total_removed = sum(s['removed'] for s in cleaning_summary.values())
    print(f"✓ Numeric cleaning complete: {len(numeric_columns)} columns, "
          f"{total_removed} invalid values removed")
    
    return df_clean

# =============================================================================
# CATEGORICAL DATA CLEANING
# =============================================================================

def standardize_categorical(df, categorical_columns=None, standardize_case=True,
                          remove_extra_spaces=True, custom_mappings=None):
    """
    Clean and standardize categorical variables.
    
    Args:
        df (DataFrame): Input DataFrame
        categorical_columns (list): List of categorical column names
        standardize_case (bool): Convert to title case
        remove_extra_spaces (bool): Remove extra whitespace
        custom_mappings (dict): Custom value mappings
        
    Returns:
        DataFrame: DataFrame with standardized categorical fields
        
    Example:
        mappings = {'state': {'calif': 'California', 'ny': 'New York'}}
        df = standardize_categorical(df, custom_mappings=mappings)
    """
    if categorical_columns is None:
        # Auto-detect categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove date and ID columns
        exclude_patterns = ['date', 'id', 'number', 'code', 'zip']
        categorical_columns = [
            col for col in categorical_columns 
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]
    
    if custom_mappings is None:
        custom_mappings = {}
    
    df_clean = df.copy()
    standardization_summary = {}
    
    for col in categorical_columns:
        if col not in df_clean.columns:
            continue
        
        original_unique = df_clean[col].nunique()
        
        # Convert to string
        df_clean[col] = df_clean[col].astype(str)
        
        # Remove extra spaces
        if remove_extra_spaces:
            df_clean[col] = df_clean[col].str.strip()
            df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # Standardize case
        if standardize_case:
            df_clean[col] = df_clean[col].str.title()
        
        # Apply custom mappings
        if col in custom_mappings:
            df_clean[col] = df_clean[col].replace(custom_mappings[col])
        
        # Replace 'nan' strings with actual NaN
        df_clean[col] = df_clean[col].replace('Nan', np.nan)
        
        final_unique = df_clean[col].nunique()
        standardization_summary[col] = {
            'original_unique': original_unique,
            'final_unique': final_unique,
            'reduction': original_unique - final_unique
        }
    
    total_reduction = sum(s['reduction'] for s in standardization_summary.values())
    print(f"✓ Categorical standardization complete: {len(categorical_columns)} columns, "
          f"{total_reduction} duplicate values consolidated")
    
    return df_clean

# =============================================================================
# COMPREHENSIVE CLEANING PIPELINE
# =============================================================================

def comprehensive_clean(df, config=None):
    """
    Apply comprehensive cleaning pipeline to DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame
        config (dict): Cleaning configuration options
        
    Returns:
        DataFrame: Fully cleaned DataFrame
        
    Example:
        clean_df = comprehensive_clean(raw_df)
    """
    if config is None:
        config = {
            'promote_headers': True,
            'standardize_columns': True,
            'clean_dates': True,
            'standardize_ids': True,
            'clean_numeric': True,
            'standardize_categorical': True,
            'remove_empty_rows': True,
            'remove_empty_columns': True
        }
    
    print("🧹 Starting comprehensive data cleaning...")
    print(f"Initial shape: {df.shape}")
    
    df_clean = df.copy()
    
    # Promote and clean headers
    if config.get('promote_headers', True):
        df_clean = promote_headers(df_clean)
    
    # Standardize column names
    if config.get('standardize_columns', True):
        df_clean = standardize_column_names(df_clean)
    
    # Clean date columns
    if config.get('clean_dates', True):
        df_clean = clean_dates(df_clean)
        df_clean = validate_date_ranges(df_clean)
    
    # Standardize ID fields
    if config.get('standardize_ids', True):
        df_clean = standardize_ids(df_clean)
    
    # Clean numeric fields
    if config.get('clean_numeric', True):
        df_clean = clean_numeric_fields(df_clean)
    
    # Standardize categorical variables
    if config.get('standardize_categorical', True):
        df_clean = standardize_categorical(df_clean)
    
    # Remove empty rows and columns
    if config.get('remove_empty_rows', True):
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(how='all')
        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            print(f"✓ Removed {removed_rows} completely empty rows")
    
    if config.get('remove_empty_columns', True):
        initial_cols = len(df_clean.columns)
        df_clean = df_clean.dropna(axis=1, how='all')
        removed_cols = initial_cols - len(df_clean.columns)
        if removed_cols > 0:
            print(f"✓ Removed {removed_cols} completely empty columns")
    
    print(f"✅ Comprehensive cleaning complete!")
    print(f"Final shape: {df_clean.shape}")
    print(f"Data reduction: {df.shape[0] - df_clean.shape[0]} rows, "
          f"{df.shape[1] - df_clean.shape[1]} columns removed")
    
    return df_clean

# =============================================================================
# SPECIALIZED CLEANING FUNCTIONS
# =============================================================================

def clean_loan_data(df):
    """
    Specialized cleaning for loan-level data from GSEs.
    
    Args:
        df (DataFrame): Raw loan data
        
    Returns:
        DataFrame: Cleaned loan data
        
    Example:
        loan_df = clean_loan_data(raw_loan_df)
    """
    print("🏠 Cleaning loan-level data...")
    
    df_clean = df.copy()
    
    # Apply comprehensive cleaning
    df_clean = comprehensive_clean(df_clean)
    
    # Loan-specific cleaning
    loan_columns = [
        'loan_sequence_number', 'origination_date', 'first_payment_date',
        'loan_amount', 'interest_rate', 'loan_term', 'ltv_ratio',
        'property_state', 'property_zip_code', 'enterprise_flag'
    ]
    
    # Ensure loan sequence numbers are strings
    if 'loan_sequence_number' in df_clean.columns:
        df_clean['loan_sequence_number'] = df_clean['loan_sequence_number'].astype(str)
    
    # Standardize enterprise flags
    if 'enterprise_flag' in df_clean.columns:
        enterprise_mapping = {
            'enterprise1': 'ENT1',
            'freddie mac': 'ENT2',
            'ent1': 'ENT1',
            'ent2': 'ENT2',
            'f': 'ENT1',
            'r': 'ENT2'
        }
        df_clean['enterprise_flag'] = df_clean['enterprise_flag'].str.lower().replace(enterprise_mapping)
    
    # Validate loan amounts
    if 'loan_amount' in df_clean.columns:
        invalid_loans = (
            (df_clean['loan_amount'] < VALIDATION_RULES['MIN_LOAN_AMOUNT']) |
            (df_clean['loan_amount'] > VALIDATION_RULES['MAX_LOAN_AMOUNT'])
        )
        if invalid_loans.any():
            print(f"⚠️  {invalid_loans.sum()} loans with invalid amounts removed")
            df_clean = df_clean[~invalid_loans]
    
    print(f"✅ Loan data cleaning complete: {len(df_clean)} loans processed")
    return df_clean

def clean_fraud_data(df):
    """
    Specialized cleaning for fraud/BSA data.
    
    Args:
        df (DataFrame): Raw fraud data
        
    Returns:
        DataFrame: Cleaned fraud data
        
    Example:
        fraud_df = clean_fraud_data(raw_fraud_df)
    """
    print("🚨 Cleaning fraud/BSA data...")
    
    df_clean = df.copy()
    
    # Apply comprehensive cleaning
    df_clean = comprehensive_clean(df_clean)
    
    # Fraud-specific cleaning
    if 'bsa_id' in df_clean.columns:
        # Remove invalid BSA IDs
        invalid_bsa = df_clean['bsa_id'].str.len() < 5
        if invalid_bsa.any():
            print(f"⚠️  {invalid_bsa.sum()} records with invalid BSA IDs removed")
            df_clean = df_clean[~invalid_bsa]
    
    # Standardize fraud types
    if 'fraud_type' in df_clean.columns:
        fraud_mapping = {
            'occupancy fraud': 'Occupancy',
            'income fraud': 'Income',
            'asset fraud': 'Asset',
            'identity fraud': 'Identity',
            'appraisal fraud': 'Appraisal'
        }
        df_clean['fraud_type'] = df_clean['fraud_type'].str.lower().replace(fraud_mapping)
    
    print(f"✅ Fraud data cleaning complete: {len(df_clean)} records processed")
    return df_clean

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_cleaning_functions():
    """
    Test cleaning functions with sample data.
    
    Returns:
        bool: True if all tests pass
    """
    print("🧪 Testing cleaning functions...")
    
    try:
        # Create sample data
        sample_data = {
            'Loan ID': ['123456', '789012', '345678'],
            'Origination Date': ['2023-01-15', '01/15/2023', '20230115'],
            'Loan Amount': ['$250,000', '300000', '275,500'],
            'State': [' california ', 'NEW YORK', 'texas'],
            'ZIP Code': ['90210', '10001', '73301']
        }
        
        test_df = pd.DataFrame(sample_data)
        print(f"✓ Created test data: {test_df.shape}")
        
        # Test header promotion
        clean_df = promote_headers(test_df)
        assert 'loan_id' in clean_df.columns.str.lower()
        print("✓ Header promotion test passed")
        
        # Test date cleaning
        clean_df = clean_dates(clean_df, ['origination_date'])
        assert clean_df['origination_date'].dtype == 'datetime64[ns]'
        print("✓ Date cleaning test passed")
        
        # Test numeric cleaning
        clean_df = clean_numeric_fields(clean_df, ['loan_amount'])
        assert clean_df['loan_amount'].dtype in ['float64', 'int64']
        print("✓ Numeric cleaning test passed")
        
        # Test categorical standardization
        clean_df = standardize_categorical(clean_df, ['state'])
        assert 'California' in clean_df['state'].values
        print("✓ Categorical standardization test passed")
        
        print("✅ All cleaning tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cleaning test failed: {str(e)}")
        return False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_cleaning_summary(df_before, df_after):
    """
    Generate summary of cleaning operations performed.
    
    Args:
        df_before (DataFrame): DataFrame before cleaning
        df_after (DataFrame): DataFrame after cleaning
        
    Returns:
        dict: Summary statistics
        
    Example:
        summary = get_cleaning_summary(raw_df, clean_df)
    """
    summary = {
        'shape_before': df_before.shape,
        'shape_after': df_after.shape,
        'rows_removed': df_before.shape[0] - df_after.shape[0],
        'columns_removed': df_before.shape[1] - df_after.shape[1],
        'data_reduction_pct': (1 - (df_after.size / df_before.size)) * 100,
        'missing_data_before': df_before.isnull().sum().sum(),
        'missing_data_after': df_after.isnull().sum().sum()
    }
    
    return summary

def export_cleaning_report(df_before, df_after, output_path=None):
    """
    Export detailed cleaning report to Excel.
    
    Args:
        df_before (DataFrame): DataFrame before cleaning
        df_after (DataFrame): DataFrame after cleaning
        output_path (Path): Output file path
        
    Returns:
        Path: Path to exported report
        
    Example:
        report_path = export_cleaning_report(raw_df, clean_df)
    """
    if output_path is None:
        from config.config import get_timestamped_path, DRAFT_DIR
        output_dir = get_timestamped_path(DRAFT_DIR)
        output_path = output_dir / 'data_cleaning_report.xlsx'
    
    # Generate summary statistics
    summary = get_cleaning_summary(df_before, df_after)
    
    # Create report DataFrames
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['Value']
    
    # Column comparison
    cols_before = set(df_before.columns)
    cols_after = set(df_after.columns)
    
    column_changes = pd.DataFrame({
        'Column': list(cols_before.union(cols_after)),
        'Before': [col in cols_before for col in cols_before.union(cols_after)],
        'After': [col in cols_after for col in cols_before.union(cols_after)],
        'Status': ['Removed' if col in cols_before and col not in cols_after
                  else 'Added' if col not in cols_before and col in cols_after
                  else 'Retained' for col in cols_before.union(cols_after)]
    })
    
    # Data quality comparison
    quality_before = df_before.isnull().sum().reset_index()
    quality_before.columns = ['Column', 'Missing_Before']
    
    quality_after = df_after.isnull().sum().reset_index() if not df_after.empty else pd.DataFrame(columns=['Column', 'Missing_After'])
    quality_after.columns = ['Column', 'Missing_After']
    
    quality_comparison = quality_before.merge(quality_after, on='Column', how='outer').fillna(0)
    quality_comparison['Improvement'] = quality_comparison['Missing_Before'] - quality_comparison['Missing_After']
    
    # Export to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary')
        column_changes.to_excel(writer, sheet_name='Column_Changes', index=False)
        quality_comparison.to_excel(writer, sheet_name='Data_Quality', index=False)
    
    print(f"📊 Cleaning report exported: {output_path}")
    return output_path

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_cleaning_functions()