"""
Data Validation and Quality Assurance Module

This module provides comprehensive data validation functions for loan-level
and fraud data, including consistency checks, range validations, and 
data quality reporting with automated flagging of problematic records.

Functions:
    validate_date_consistency: Check for logical date relationships
    validate_numeric_ranges: Ensure numeric values within acceptable bounds
    flag_invalid_records: Identify records with critical data issues
    generate_quality_report: Comprehensive data quality assessment
    validate_loan_data: Specialized validation for loan records
    validate_fraud_data: Specialized validation for fraud/BSA records

Example:
    from modules.validation import validate_loan_data, generate_quality_report
    
    # Validate loan data
    validation_results = validate_loan_data(loan_df)
    
    # Generate quality report
    quality_report = generate_quality_report(clean_df)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import VALIDATION_RULES, ENTERPRISE_COLORS

# =============================================================================
# DATE VALIDATION FUNCTIONS
# =============================================================================

def validate_date_consistency(df, date_relationships=None, tolerance_days=0):
    """
    Validate logical relationships between date fields.
    
    Args:
        df (DataFrame): Input DataFrame with date columns
        date_relationships (list): List of date relationship tuples (earlier, later, description)
        tolerance_days (int): Allowable tolerance in days for date comparisons
        
    Returns:
        dict: Validation results with flagged records
        
    Example:
        relationships = [('origination_date', 'first_payment_date', 'First payment after origination')]
        results = validate_date_consistency(df, relationships)
    """
    if date_relationships is None:
        date_relationships = [
            ('origination_date', 'first_payment_date', 'First payment after origination'),
            ('origination_date', 'maturity_date', 'Maturity after origination'),
            ('first_payment_date', 'maturity_date', 'Maturity after first payment')
        ]
    
    validation_results = {
        'total_records': len(df),
        'date_issues': {},
        'flagged_records': set(),
        'summary': {}
    }
    
    for earlier_col, later_col, description in date_relationships:
        if earlier_col not in df.columns or later_col not in df.columns:
            continue
        
        # Check for records where dates are available
        valid_dates_mask = df[earlier_col].notna() & df[later_col].notna()
        
        if not valid_dates_mask.any():
            continue
        
        # Check date logic with tolerance
        date_diff = (df[later_col] - df[earlier_col]).dt.days
        invalid_mask = valid_dates_mask & (date_diff < -tolerance_days)
        
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            invalid_indices = df[invalid_mask].index.tolist()
            validation_results['flagged_records'].update(invalid_indices)
            
            validation_results['date_issues'][f"{earlier_col}_vs_{later_col}"] = {
                'description': description,
                'invalid_count': invalid_count,
                'invalid_percentage': (invalid_count / valid_dates_mask.sum()) * 100,
                'invalid_indices': invalid_indices,
                'min_diff_days': date_diff[invalid_mask].min(),
                'max_diff_days': date_diff[invalid_mask].max()
            }
    
    # Generate summary
    total_flagged = len(validation_results['flagged_records'])
    validation_results['summary'] = {
        'total_date_issues': len(validation_results['date_issues']),
        'total_flagged_records': total_flagged,
        'flagged_percentage': (total_flagged / len(df)) * 100 if len(df) > 0 else 0
    }
    
    if total_flagged > 0:
        print(f"⚠️  Date validation: {total_flagged} records flagged for date inconsistencies")
    else:
        print("✅ Date validation: All date relationships are logical")
    
    return validation_results

def validate_date_ranges(df, date_columns=None, business_rules=None):
    """
    Validate that dates fall within business-logical ranges.
    
    Args:
        df (DataFrame): Input DataFrame
        date_columns (list): Date columns to validate
        business_rules (dict): Custom business rules for date ranges
        
    Returns:
        dict: Validation results with out-of-range dates
        
    Example:
        rules = {'origination_date': {'min': '1990-01-01', 'max': '2025-12-31'}}
        results = validate_date_ranges(df, business_rules=rules)
    """
    if date_columns is None:
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    
    if business_rules is None:
        current_year = datetime.now().year
        business_rules = {
            'origination_date': {
                'min': f'{current_year - 50}-01-01',
                'max': f'{current_year}-12-31'
            },
            'maturity_date': {
                'min': f'{current_year - 50}-01-01',
                'max': f'{current_year + 50}-12-31'
            },
            'first_payment_date': {
                'min': f'{current_year - 50}-01-01',
                'max': f'{current_year + 1}-12-31'
            }
        }
    
    validation_results = {
        'total_records': len(df),
        'range_violations': {},
        'flagged_records': set(),
        'summary': {}
    }
    
    for col in date_columns:
        if col not in df.columns or col not in business_rules:
            continue
        
        rules = business_rules[col]
        min_date = pd.to_datetime(rules['min'])
        max_date = pd.to_datetime(rules['max'])
        
        # Check for valid dates
        valid_dates_mask = df[col].notna()
        
        if not valid_dates_mask.any():
            continue
        
        # Check range violations
        too_early = valid_dates_mask & (df[col] < min_date)
        too_late = valid_dates_mask & (df[col] > max_date)
        
        violations = too_early | too_late
        violation_count = violations.sum()
        
        if violation_count > 0:
            violation_indices = df[violations].index.tolist()
            validation_results['flagged_records'].update(violation_indices)
            
            validation_results['range_violations'][col] = {
                'total_violations': violation_count,
                'too_early_count': too_early.sum(),
                'too_late_count': too_late.sum(),
                'violation_percentage': (violation_count / valid_dates_mask.sum()) * 100,
                'violation_indices': violation_indices,
                'min_allowed': min_date,
                'max_allowed': max_date,
                'earliest_violation': df.loc[too_early, col].min() if too_early.any() else None,
                'latest_violation': df.loc[too_late, col].max() if too_late.any() else None
            }
    
    # Generate summary
    total_flagged = len(validation_results['flagged_records'])
    validation_results['summary'] = {
        'total_range_violations': len(validation_results['range_violations']),
        'total_flagged_records': total_flagged,
        'flagged_percentage': (total_flagged / len(df)) * 100 if len(df) > 0 else 0
    }
    
    if total_flagged > 0:
        print(f"⚠️  Date range validation: {total_flagged} records with dates outside business ranges")
    else:
        print("✅ Date range validation: All dates within acceptable ranges")
    
    return validation_results

# =============================================================================
# NUMERIC VALIDATION FUNCTIONS
# =============================================================================

def validate_numeric_ranges(df, numeric_rules=None, flag_outliers=True, outlier_method='iqr'):
    """
    Validate numeric fields against business rules and statistical outliers.
    
    Args:
        df (DataFrame): Input DataFrame
        numeric_rules (dict): Business rules for numeric validation
        flag_outliers (bool): Whether to flag statistical outliers
        outlier_method (str): Method for outlier detection ('iqr', 'zscore')
        
    Returns:
        dict: Validation results with range violations and outliers
        
    Example:
        rules = {'loan_amount': {'min': 1000, 'max': 5000000}}
        results = validate_numeric_ranges(df, rules)
    """
    if numeric_rules is None:
        numeric_rules = {
            'loan_amount': {
                'min': VALIDATION_RULES['MIN_LOAN_AMOUNT'],
                'max': VALIDATION_RULES['MAX_LOAN_AMOUNT']
            },
            'ltv_ratio': {'min': 0, 'max': 200},
            'dti_ratio': {'min': 0, 'max': 100},
            'credit_score': {'min': 300, 'max': 850},
            'interest_rate': {'min': 0, 'max': 30}
        }
    
    validation_results = {
        'total_records': len(df),
        'range_violations': {},
        'outliers': {},
        'flagged_records': set(),
        'summary': {}
    }
    
    for col, rules in numeric_rules.items():
        if col not in df.columns:
            continue
        
        # Check for valid numeric values
        valid_numeric_mask = df[col].notna() & np.isfinite(df[col])
        
        if not valid_numeric_mask.any():
            continue
        
        # Range validation
        min_val = rules.get('min', -np.inf)
        max_val = rules.get('max', np.inf)
        
        below_min = valid_numeric_mask & (df[col] < min_val)
        above_max = valid_numeric_mask & (df[col] > max_val)
        range_violations = below_min | above_max
        
        if range_violations.any():
            violation_indices = df[range_violations].index.tolist()
            validation_results['flagged_records'].update(violation_indices)
            
            validation_results['range_violations'][col] = {
                'total_violations': range_violations.sum(),
                'below_min_count': below_min.sum(),
                'above_max_count': above_max.sum(),
                'violation_percentage': (range_violations.sum() / valid_numeric_mask.sum()) * 100,
                'violation_indices': violation_indices,
                'min_allowed': min_val,
                'max_allowed': max_val,
                'min_violation': df.loc[below_min, col].min() if below_min.any() else None,
                'max_violation': df.loc[above_max, col].max() if above_max.any() else None
            }
        
        # Outlier detection
        if flag_outliers:
            outlier_mask = _detect_outliers(df[col][valid_numeric_mask], method=outlier_method)
            
            if outlier_mask.any():
                # Map back to original DataFrame indices
                outlier_indices = df[valid_numeric_mask].index[outlier_mask].tolist()
                validation_results['flagged_records'].update(outlier_indices)
                
                validation_results['outliers'][col] = {
                    'outlier_count': outlier_mask.sum(),
                    'outlier_percentage': (outlier_mask.sum() / valid_numeric_mask.sum()) * 100,
                    'outlier_indices': outlier_indices,
                    'method': outlier_method,
                    'min_outlier': df.loc[outlier_indices, col].min() if outlier_indices else None,
                    'max_outlier': df.loc[outlier_indices, col].max() if outlier_indices else None
                }
    
    # Generate summary
    total_flagged = len(validation_results['flagged_records'])
    validation_results['summary'] = {
        'total_range_violations': len(validation_results['range_violations']),
        'total_outlier_fields': len(validation_results['outliers']),
        'total_flagged_records': total_flagged,
        'flagged_percentage': (total_flagged / len(df)) * 100 if len(df) > 0 else 0
    }
    
    if total_flagged > 0:
        print(f"⚠️  Numeric validation: {total_flagged} records flagged for range violations or outliers")
    else:
        print("✅ Numeric validation: All numeric values within acceptable ranges")
    
    return validation_results

def _detect_outliers(series, method='iqr', z_threshold=3):
    """
    Detect outliers in numeric series using specified method.
    
    Args:
        series (Series): Numeric series to analyze
        method (str): Detection method ('iqr', 'zscore')
        z_threshold (float): Z-score threshold for outlier detection
        
    Returns:
        Series: Boolean mask indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > z_threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

# =============================================================================
# RECORD-LEVEL VALIDATION
# =============================================================================

def flag_invalid_records(df, validation_rules=None, severity_levels=None):
    """
    Flag records with critical data quality issues.
    
    Args:
        df (DataFrame): Input DataFrame
        validation_rules (dict): Custom validation rules
        severity_levels (dict): Severity classification for different issues
        
    Returns:
        DataFrame: Original DataFrame with validation flags added
        
    Example:
        flagged_df = flag_invalid_records(loan_df)
    """
    if validation_rules is None:
        validation_rules = {
            'missing_critical_fields': ['loan_sequence_number', 'origination_date', 'loan_amount'],
            'invalid_date_logic': [('origination_date', 'maturity_date')],
            'suspicious_values': {
                'loan_amount': {'min': 1000, 'max': 50000000},
                'ltv_ratio': {'min': 0, 'max': 200}
            }
        }
    
    if severity_levels is None:
        severity_levels = {
            'CRITICAL': 'Record unusable for analysis',
            'WARNING': 'Record may have data quality issues',
            'INFO': 'Minor data quality note'
        }
    
    df_flagged = df.copy()
    
    # Initialize validation flag columns
    df_flagged['validation_flags'] = ''
    df_flagged['validation_severity'] = 'VALID'
    df_flagged['validation_issues'] = ''
    
    issues_found = {
        'CRITICAL': 0,
        'WARNING': 0,
        'INFO': 0
    }
    
    # Check for missing critical fields
    critical_fields = validation_rules.get('missing_critical_fields', [])
    for field in critical_fields:
        if field in df_flagged.columns:
            missing_mask = df_flagged[field].isna()
            if missing_mask.any():
                df_flagged.loc[missing_mask, 'validation_flags'] += f'MISSING_{field.upper()};'
                df_flagged.loc[missing_mask, 'validation_severity'] = 'CRITICAL'
                df_flagged.loc[missing_mask, 'validation_issues'] += f'Missing {field}; '
                issues_found['CRITICAL'] += missing_mask.sum()
    
    # Check date logic
    date_logic_rules = validation_rules.get('invalid_date_logic', [])
    for earlier_col, later_col in date_logic_rules:
        if earlier_col in df_flagged.columns and later_col in df_flagged.columns:
            valid_dates = df_flagged[earlier_col].notna() & df_flagged[later_col].notna()
            invalid_logic = valid_dates & (df_flagged[earlier_col] >= df_flagged[later_col])
            
            if invalid_logic.any():
                df_flagged.loc[invalid_logic, 'validation_flags'] += f'INVALID_DATE_LOGIC_{earlier_col}_{later_col};'
                df_flagged.loc[invalid_logic, 'validation_severity'] = 'WARNING'
                df_flagged.loc[invalid_logic, 'validation_issues'] += f'Invalid date logic: {earlier_col} >= {later_col}; '
                issues_found['WARNING'] += invalid_logic.sum()
    
    # Check suspicious values
    suspicious_rules = validation_rules.get('suspicious_values', {})
    for col, rules in suspicious_rules.items():
        if col in df_flagged.columns:
            valid_values = df_flagged[col].notna()
            
            if 'min' in rules:
                below_min = valid_values & (df_flagged[col] < rules['min'])
                if below_min.any():
                    df_flagged.loc[below_min, 'validation_flags'] += f'SUSPICIOUS_{col.upper()}_LOW;'
                    df_flagged.loc[below_min, 'validation_issues'] += f'{col} below minimum; '
                    
                    # Set severity if not already critical
                    current_severity = df_flagged.loc[below_min, 'validation_severity']
                    new_severity = np.where(current_severity == 'VALID', 'WARNING', current_severity)
                    df_flagged.loc[below_min, 'validation_severity'] = new_severity
                    issues_found['WARNING'] += below_min.sum()
            
            if 'max' in rules:
                above_max = valid_values & (df_flagged[col] > rules['max'])
                if above_max.any():
                    df_flagged.loc[above_max, 'validation_flags'] += f'SUSPICIOUS_{col.upper()}_HIGH;'
                    df_flagged.loc[above_max, 'validation_issues'] += f'{col} above maximum; '
                    
                    # Set severity if not already critical
                    current_severity = df_flagged.loc[above_max, 'validation_severity']
                    new_severity = np.where(current_severity == 'VALID', 'WARNING', current_severity)
                    df_flagged.loc[above_max, 'validation_severity'] = new_severity
                    issues_found['WARNING'] += above_max.sum()
    
    # Clean up flag columns
    df_flagged['validation_flags'] = df_flagged['validation_flags'].str.rstrip(';')
    df_flagged['validation_issues'] = df_flagged['validation_issues'].str.rstrip('; ')
    
    # Print summary
    total_issues = sum(issues_found.values())
    if total_issues > 0:
        print(f"🚩 Record validation complete:")
        for severity, count in issues_found.items():
            if count > 0:
                print(f"   {severity}: {count} records")
    else:
        print("✅ Record validation: No data quality issues found")
    
    return df_flagged

# =============================================================================
# COMPREHENSIVE QUALITY REPORTING
# =============================================================================

def generate_quality_report(df, include_profiling=True, export_path=None):
    """
    Generate comprehensive data quality assessment report.
    
    Args:
        df (DataFrame): Input DataFrame to assess
        include_profiling (bool): Include detailed data profiling
        export_path (Path): Optional path to export report
        
    Returns:
        dict: Comprehensive quality report
        
    Example:
        quality_report = generate_quality_report(clean_df)
    """
    print("📊 Generating comprehensive data quality report...")
    
    quality_report = {
        'metadata': {
            'report_timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        'completeness': {},
        'consistency': {},
        'validity': {},
        'profiling': {},
        'recommendations': []
    }
    
    # Completeness Assessment
    quality_report['completeness'] = _assess_completeness(df)
    
    # Consistency Assessment
    quality_report['consistency'] = _assess_consistency(df)
    
    # Validity Assessment
    quality_report['validity'] = _assess_validity(df)
    
    # Data Profiling
    if include_profiling:
        quality_report['profiling'] = _generate_data_profile(df)
    
    # Generate Recommendations
    quality_report['recommendations'] = _generate_recommendations(quality_report)
    
    # Export report if requested
    if export_path:
        _export_quality_report(quality_report, export_path)
    
    # Print summary
    _print_quality_summary(quality_report)
    
    return quality_report

def _assess_completeness(df):
    """Assess data completeness across all columns."""
    completeness = {
        'overall_completeness': (1 - df.isnull().sum().sum() / df.size) * 100,
        'column_completeness': {},
        'critical_missing': [],
        'completeness_by_column': {}
    }
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        total_count = len(df)
        completeness_pct = ((total_count - missing_count) / total_count) * 100
        
        completeness['column_completeness'][col] = {
            'missing_count': missing_count,
            'completeness_percentage': completeness_pct,
            'data_type': str(df[col].dtype)
        }
        
        # Flag critical missing data (>50% missing)
        if completeness_pct < 50:
            completeness['critical_missing'].append(col)
    
    return completeness

def _assess_consistency(df):
    """Assess data consistency and standardization."""
    consistency = {
        'duplicate_records': len(df) - len(df.drop_duplicates()),
        'data_type_consistency': {},
        'format_consistency': {},
        'categorical_consistency': {}
    }
    
    # Check data type consistency
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object':
            # Check for mixed types in object columns
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                type_counts = non_null_values.apply(type).value_counts()
                consistency['data_type_consistency'][col] = {
                    'primary_type': str(type_counts.index[0].__name__),
                    'type_distribution': {str(t.__name__): count for t, count in type_counts.items()},
                    'is_consistent': len(type_counts) == 1
                }
    
    # Check format consistency for common patterns
    for col in df.select_dtypes(include=['object']).columns:
        if 'date' in col.lower():
            consistency['format_consistency'][col] = _check_date_formats(df[col])
        elif any(pattern in col.lower() for pattern in ['zip', 'phone', 'ssn', 'id']):
            consistency['format_consistency'][col] = _check_id_formats(df[col])
    
    # Categorical consistency
    for col in df.select_dtypes(include=['object']).columns:
        unique_values = df[col].nunique()
        total_values = df[col].notna().sum()
        
        if unique_values / total_values < 0.1:  # Likely categorical
            value_counts = df[col].value_counts()
            consistency['categorical_consistency'][col] = {
                'unique_values': unique_values,
                'most_common': value_counts.head(5).to_dict(),
                'potential_duplicates': _find_similar_categories(value_counts.index)
            }
    
    return consistency

def _assess_validity(df):
    """Assess data validity against business rules."""
    validity = {
        'range_violations': {},
        'pattern_violations': {},
        'logical_violations': {},
        'outlier_analysis': {}
    }
    
    # Numeric range validation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['loan_amount', 'property_value']:
            violations = (df[col] < 0) | (df[col] > 100000000)
        elif 'ratio' in col.lower() or 'rate' in col.lower():
            violations = (df[col] < 0) | (df[col] > 100)
        else:
            violations = pd.Series([False] * len(df))
        
        if violations.any():
            validity['range_violations'][col] = {
                'violation_count': violations.sum(),
                'violation_percentage': (violations.sum() / len(df)) * 100
            }
    
    # Pattern validation for common fields
    pattern_rules = {
        'zip_code': r'^\d{5}(-\d{4})?,
        'phone': r'^\d{10}$|^\d{3}-\d{3}-\d{4},
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
    }
    
    for col in df.columns:
        for pattern_name, pattern in pattern_rules.items():
            if pattern_name in col.lower():
                non_null_values = df[col].dropna().astype(str)
                if len(non_null_values) > 0:
                    invalid_pattern = ~non_null_values.str.match(pattern)
                    if invalid_pattern.any():
                        validity['pattern_violations'][col] = {
                            'pattern': pattern,
                            'violation_count': invalid_pattern.sum(),
                            'violation_percentage': (invalid_pattern.sum() / len(non_null_values)) * 100
                        }
    
    return validity

def _generate_data_profile(df):
    """Generate detailed data profiling information."""
    profiling = {
        'numeric_summary': {},
        'categorical_summary': {},
        'date_summary': {},
        'correlation_analysis': {}
    }
    
    # Numeric profiling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profiling['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            # Find high correlations (>0.8)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            profiling['correlation_analysis']['high_correlations'] = high_corr_pairs
    
    # Categorical profiling
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() / len(df) < 0.5:  # Likely categorical
            profiling['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'value_counts': df[col].value_counts().head(10).to_dict()
            }
    
    # Date profiling
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in date_cols:
        profiling['date_summary'][col] = {
            'min_date': df[col].min(),
            'max_date': df[col].max(),
            'date_range_days': (df[col].max() - df[col].min()).days if df[col].notna().any() else 0,
            'most_common_year': df[col].dt.year.mode().iloc[0] if df[col].notna().any() else None
        }
    
    return profiling

def _generate_recommendations(quality_report):
    """Generate actionable recommendations based on quality assessment."""
    recommendations = []
    
    # Completeness recommendations
    completeness = quality_report['completeness']
    if completeness['overall_completeness'] < 90:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Completeness',
            'issue': f"Overall data completeness is {completeness['overall_completeness']:.1f}%",
            'recommendation': 'Investigate data collection processes and consider imputation strategies'
        })
    
    for col in completeness['critical_missing']:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Completeness',
            'issue': f"Column '{col}' has >50% missing values",
            'recommendation': f'Consider removing column {col} or improving data collection'
        })
    
    # Consistency recommendations
    consistency = quality_report['consistency']
    if consistency['duplicate_records'] > 0:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Consistency',
            'issue': f"{consistency['duplicate_records']} duplicate records found",
            'recommendation': 'Remove duplicate records and implement deduplication logic'
        })
    
    # Validity recommendations
    validity = quality_report['validity']
    for col, violation_info in validity['range_violations'].items():
        if violation_info['violation_percentage'] > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Validity',
                'issue': f"Column '{col}' has {violation_info['violation_percentage']:.1f}% range violations",
                'recommendation': f'Review business rules for {col} and clean invalid values'
            })
    
    return recommendations

def _check_date_formats(series):
    """Check consistency of date formats in a series."""
    non_null_values = series.dropna().astype(str)
    if len(non_null_values) == 0:
        return {'format_consistency': True, 'patterns': []}
    
    # Common date patterns
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2},  # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4},  # MM/DD/YYYY
        r'^\d{2}-\d{2}-\d{4},  # MM-DD-YYYY
        r'^\d{8}               # YYYYMMDD
    ]
    
    pattern_matches = {}
    for pattern in date_patterns:
        matches = non_null_values.str.match(pattern).sum()
        if matches > 0:
            pattern_matches[pattern] = matches
    
    return {
        'format_consistency': len(pattern_matches) <= 1,
        'patterns': pattern_matches,
        'dominant_pattern': max(pattern_matches.items(), key=lambda x: x[1])[0] if pattern_matches else None
    }

def _check_id_formats(series):
    """Check consistency of ID formats in a series."""
    non_null_values = series.dropna().astype(str)
    if len(non_null_values) == 0:
        return {'format_consistency': True, 'length_distribution': {}}
    
    length_distribution = non_null_values.str.len().value_counts().to_dict()
    
    return {
        'format_consistency': len(length_distribution) <= 2,
        'length_distribution': length_distribution,
        'dominant_length': max(length_distribution.items(), key=lambda x: x[1])[0] if length_distribution else None
    }

def _find_similar_categories(categories):
    """Find potentially duplicate categorical values."""
    similar_pairs = []
    categories_list = list(categories)
    
    for i, cat1 in enumerate(categories_list):
        for cat2 in categories_list[i+1:]:
            if isinstance(cat1, str) and isinstance(cat2, str):
                # Simple similarity check
                if cat1.lower().strip() == cat2.lower().strip() and cat1 != cat2:
                    similar_pairs.append((cat1, cat2))
    
    return similar_pairs

def _export_quality_report(quality_report, export_path):
    """Export quality report to Excel file."""
    with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': ['Total Records', 'Total Columns', 'Overall Completeness (%)', 
                      'Duplicate Records', 'Critical Missing Columns'],
            'Value': [
                quality_report['metadata']['total_records'],
                quality_report['metadata']['total_columns'],
                round(quality_report['completeness']['overall_completeness'], 2),
                quality_report['consistency']['duplicate_records'],
                len(quality_report['completeness']['critical_missing'])
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Completeness details
        completeness_df = pd.DataFrame(quality_report['completeness']['column_completeness']).T
        completeness_df.to_excel(writer, sheet_name='Completeness')
        
        # Recommendations
        if quality_report['recommendations']:
            recommendations_df = pd.DataFrame(quality_report['recommendations'])
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
    
    print(f"📋 Quality report exported: {export_path}")

def _print_quality_summary(quality_report):
    """Print quality report summary to console."""
    print("\n📊 Data Quality Summary")
    print("=" * 50)
    
    metadata = quality_report['metadata']
    completeness = quality_report['completeness']
    consistency = quality_report['consistency']
    
    print(f"Dataset Size: {metadata['total_records']:,} records × {metadata['total_columns']} columns")
    print(f"Overall Completeness: {completeness['overall_completeness']:.1f}%")
    print(f"Duplicate Records: {consistency['duplicate_records']:,}")
    
    # Critical issues
    critical_issues = len(completeness['critical_missing'])
    if critical_issues > 0:
        print(f"⚠️  Critical Issues: {critical_issues} columns with >50% missing data")
    
    # Recommendations summary
    recommendations = quality_report['recommendations']
    if recommendations:
        priority_counts = {}
        for rec in recommendations:
            priority = rec['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        print("\n🎯 Recommendations by Priority:")
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if priority in priority_counts:
                print(f"   {priority}: {priority_counts[priority]} items")
    else:
        print("✅ No critical data quality issues identified")

# =============================================================================
# SPECIALIZED VALIDATION FUNCTIONS
# =============================================================================

def validate_loan_data(df, loan_validation_rules=None):
    """
    Comprehensive validation specifically for loan-level data.
    
    Args:
        df (DataFrame): Loan data DataFrame
        loan_validation_rules (dict): Custom loan validation rules
        
    Returns:
        dict: Loan-specific validation results
        
    Example:
        loan_results = validate_loan_data(loan_df)
    """
    print("🏠 Validating loan-level data...")
    
    if loan_validation_rules is None:
        loan_validation_rules = {
            'required_fields': [
                'loan_sequence_number', 'origination_date', 'loan_amount',
                'interest_rate', 'loan_term', 'property_state'
            ],
            'date_relationships': [
                ('origination_date', 'first_payment_date', 'First payment after origination'),
                ('origination_date', 'maturity_date', 'Maturity after origination')
            ],
            'business_rules': {
                'loan_amount': {'min': 1000, 'max': 50000000},
                'interest_rate': {'min': 0.1, 'max': 30.0},
                'loan_term': {'min': 1, 'max': 480},  # months
                'ltv_ratio': {'min': 0, 'max': 200}
            }
        }
    
    validation_results = {
        'loan_validation_summary': {},
        'missing_required_fields': {},
        'date_validation': {},
        'business_rule_validation': {},
        'enterprise_distribution': {},
        'flagged_loans': set()
    }
    
    # Check required fields
    required_fields = loan_validation_rules['required_fields']
    for field in required_fields:
        if field in df.columns:
            missing_count = df[field].isnull().sum()
            if missing_count > 0:
                validation_results['missing_required_fields'][field] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(df)) * 100
                }
                validation_results['flagged_loans'].update(df[df[field].isnull()].index.tolist())
    
    # Date validation
    date_relationships = loan_validation_rules['date_relationships']
    validation_results['date_validation'] = validate_date_consistency(df, date_relationships)
    validation_results['flagged_loans'].update(validation_results['date_validation']['flagged_records'])
    
    # Business rule validation
    business_rules = loan_validation_rules['business_rules']
    validation_results['business_rule_validation'] = validate_numeric_ranges(df, business_rules)
    validation_results['flagged_loans'].update(validation_results['business_rule_validation']['flagged_records'])
    
    # Enterprise distribution analysis
    if 'enterprise_flag' in df.columns:
        enterprise_counts = df['enterprise_flag'].value_counts()
        validation_results['enterprise_distribution'] = {
            'total_loans': len(df),
            'enterprise_counts': enterprise_counts.to_dict(),
            'enterprise_percentages': (enterprise_counts / len(df) * 100).to_dict()
        }
    
    # Generate summary
    total_flagged = len(validation_results['flagged_loans'])
    validation_results['loan_validation_summary'] = {
        'total_loans': len(df),
        'flagged_loans': total_flagged,
        'flagged_percentage': (total_flagged / len(df)) * 100 if len(df) > 0 else 0,
        'missing_field_issues': len(validation_results['missing_required_fields']),
        'date_issues': validation_results['date_validation']['summary']['total_date_issues'],
        'business_rule_violations': len(validation_results['business_rule_validation']['range_violations'])
    }
    
    print(f"✅ Loan validation complete: {total_flagged}/{len(df)} loans flagged")
    
    return validation_results

def validate_fraud_data(df, fraud_validation_rules=None):
    """
    Comprehensive validation specifically for fraud/BSA data.
    
    Args:
        df (DataFrame): Fraud data DataFrame
        fraud_validation_rules (dict): Custom fraud validation rules
        
    Returns:
        dict: Fraud-specific validation results
        
    Example:
        fraud_results = validate_fraud_data(fraud_df)
    """
    print("🚨 Validating fraud/BSA data...")
    
    if fraud_validation_rules is None:
        fraud_validation_rules = {
            'required_fields': [
                'bsa_id', 'report_date', 'fraud_type', 'amount'
            ],
            'bsa_id_pattern': r'^BSA\d{8,},
            'valid_fraud_types': [
                'Occupancy', 'Income', 'Asset', 'Identity', 'Appraisal', 'Other'
            ],
            'amount_thresholds': {
                'suspicious_low': 100,
                'suspicious_high': 10000000
            }
        }
    
    validation_results = {
        'fraud_validation_summary': {},
        'bsa_id_validation': {},
        'fraud_type_validation': {},
        'amount_validation': {},
        'flagged_records': set()
    }
    
    # BSA ID validation
    if 'bsa_id' in df.columns:
        bsa_pattern = fraud_validation_rules['bsa_id_pattern']
        valid_bsa_mask = df['bsa_id'].astype(str).str.match(bsa_pattern, na=False)
        invalid_bsa_count = (~valid_bsa_mask).sum()
        
        validation_results['bsa_id_validation'] = {
            'total_records': len(df),
            'valid_bsa_ids': valid_bsa_mask.sum(),
            'invalid_bsa_ids': invalid_bsa_count,
            'invalid_percentage': (invalid_bsa_count / len(df)) * 100
        }
        
        if invalid_bsa_count > 0:
            validation_results['flagged_records'].update(df[~valid_bsa_mask].index.tolist())
    
    # Fraud type validation
    if 'fraud_type' in df.columns:
        valid_types = fraud_validation_rules['valid_fraud_types']
        invalid_types = ~df['fraud_type'].isin(valid_types + [np.nan])
        invalid_type_count = invalid_types.sum()
        
        validation_results['fraud_type_validation'] = {
            'valid_fraud_types': valid_types,
            'invalid_types_count': invalid_type_count,
            'invalid_types': df.loc[invalid_types, 'fraud_type'].unique().tolist(),
            'type_distribution': df['fraud_type'].value_counts().to_dict()
        }
        
        if invalid_type_count > 0:
            validation_results['flagged_records'].update(df[invalid_types].index.tolist())
    
    # Amount validation
    if 'amount' in df.columns:
        thresholds = fraud_validation_rules['amount_thresholds']
        suspicious_low = df['amount'] < thresholds['suspicious_low']
        suspicious_high = df['amount'] > thresholds['suspicious_high']
        
        validation_results['amount_validation'] = {
            'suspicious_low_count': suspicious_low.sum(),
            'suspicious_high_count': suspicious_high.sum(),
            'amount_statistics': df['amount'].describe().to_dict()
        }
        
        # Flag suspicious amounts (but not necessarily invalid)
        suspicious_amounts = suspicious_low | suspicious_high
        if suspicious_amounts.any():
            validation_results['flagged_records'].update(df[suspicious_amounts].index.tolist())
    
    # Generate summary
    total_flagged = len(validation_results['flagged_records'])
    validation_results['fraud_validation_summary'] = {
        'total_records': len(df),
        'flagged_records': total_flagged,
        'flagged_percentage': (total_flagged / len(df)) * 100 if len(df) > 0 else 0
    }
    
    print(f"✅ Fraud validation complete: {total_flagged}/{len(df)} records flagged")
    
    return validation_results

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_validation_functions():
    """
    Test validation functions with sample data.
    
    Returns:
        bool: True if all tests pass
    """
    print("🧪 Testing validation functions...")
    
    try:
        # Create sample loan data
        sample_loan_data = {
            'loan_sequence_number': ['123456', '789012', '345678'],
            'origination_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10']),
            'first_payment_date': pd.to_datetime(['2023-02-15', '2023-03-20', '2023-04-10']),
            'maturity_date': pd.to_datetime(['2053-01-15', '2053-02-20', '2053-03-10']),
            'loan_amount': [250000, 350000, 180000],
            'interest_rate': [3.5, 4.2, 3.8],
            'enterprise_flag': ['ENT1', 'ENT2', 'ENT1']
        }
        
        test_df = pd.DataFrame(sample_loan_data)
        print(f"✓ Created test loan data: {test_df.shape}")
        
        # Test date consistency validation
        date_results = validate_date_consistency(test_df)
        assert 'summary' in date_results
        print("✓ Date consistency validation test passed")
        
        # Test numeric range validation
        numeric_results = validate_numeric_ranges(test_df)
        assert 'summary' in numeric_results
        print("✓ Numeric range validation test passed")
        
        # Test record flagging
        flagged_df = flag_invalid_records(test_df)
        assert 'validation_flags' in flagged_df.columns
        print("✓ Record flagging test passed")
        
        # Test quality report generation
        quality_report = generate_quality_report(test_df, include_profiling=False)
        assert 'metadata' in quality_report
        print("✓ Quality report generation test passed")
        
        # Test loan-specific validation
        loan_results = validate_loan_data(test_df)
        assert 'loan_validation_summary' in loan_results
        print("✓ Loan data validation test passed")
        
        print("✅ All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_validation_functions()