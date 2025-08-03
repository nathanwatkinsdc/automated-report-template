"""
Standardized Visualization Functions Module

This module provides reusable plotting functions with consistent styling
for automated regulatory reporting. Includes choropleth maps, time series,
bar charts, and other common visualizations using Plotly for interactivity
and matplotlib for static exports.

Functions:
    plot_time_series: Time-based trend analysis with dual axes support
    plot_choropleth: Geographic visualizations normalized by exposure
    plot_horizontal_bars: Comparative horizontal bar charts
    plot_loan_volume_trends: Specialized loan volume analysis
    plot_fraud_heatmap: Geographic fraud concentration mapping
    create_dashboard_layout: Multi-panel dashboard arrangements

Example:
    from modules.visuals import plot_time_series, plot_choropleth
    
    # Create time series plot
    fig = plot_time_series(df, x_col='date', y_col='volume', 
                          title='Loan Volume Trends')
    
    # Create choropleth map
    map_fig = plot_choropleth(df, location_col='state', 
                             value_col='fraud_count')
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.config import ENTERPRISE_COLORS, SEQUENTIAL_COLORS, DIVERGING_COLORS, DEFAULT_FIGURE_SIZE

# =============================================================================
# TIME SERIES VISUALIZATIONS
# =============================================================================

def plot_time_series(df, x_col, y_col, group_col=None, title=None, 
                    secondary_y_col=None, show_trend=False, date_format='%Y-%m'):
    """
    Create interactive time series plot with optional secondary axis.
    
    Args:
        df (DataFrame): Input data
        x_col (str): Date/time column name
        y_col (str): Primary y-axis variable
        group_col (str): Optional grouping column (e.g., enterprise)
        title (str): Chart title
        secondary_y_col (str): Optional secondary y-axis variable
        show_trend (bool): Add trend line
        date_format (str): Date format for x-axis labels
        
    Returns:
        plotly.graph_objects.Figure: Interactive time series plot
        
    Example:
        fig = plot_time_series(df, 'reporting_period', 'loan_count', 
                              group_col='enterprise_flag', 
                              title='Monthly Loan Originations')
    """
    if title is None:
        title = f'{y_col.replace("_", " ").title()} Over Time'
    
    # Create figure with secondary y-axis if needed
    if secondary_y_col:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    
    # Ensure x column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df[x_col] = pd.to_datetime(df[x_col])
    
    # Plot primary series
    if group_col and group_col in df.columns:
        # Group by specified column
        for i, (group, group_df) in enumerate(df.groupby(group_col)):
            color = ENTERPRISE_COLORS.get(group, px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)])
            
            # Sort by date for proper line plotting
            group_df = group_df.sort_values(x_col)
            
            if secondary_y_col:
                fig.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df[y_col],
                        mode='lines+markers',
                        name=f'{group} - {y_col}',
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    ),
                    secondary_y=False
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df[y_col],
                        mode='lines+markers',
                        name=f'{group}' if len(df[group_col].unique()) > 1 else y_col,
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    )
                )
            
            # Add trend line if requested
            if show_trend:
                z = np.polyfit(range(len(group_df)), group_df[y_col], 1)
                trend_line = np.poly1d(z)(range(len(group_df)))
                
                fig.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=trend_line,
                        mode='lines',
                        name=f'{group} Trend',
                        line=dict(color=color, width=1, dash='dash'),
                        showlegend=False
                    )
                )
    else:
        # Single series
        df_sorted = df.sort_values(x_col)
        
        if secondary_y_col:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[x_col],
                    y=df_sorted[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=ENTERPRISE_COLORS['NEUTRAL'], width=2),
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[x_col],
                    y=df_sorted[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=ENTERPRISE_COLORS['NEUTRAL'], width=2),
                    marker=dict(size=6)
                )
            )
    
    # Add secondary y-axis data if provided
    if secondary_y_col and secondary_y_col in df.columns:
        if group_col and group_col in df.columns:
            for i, (group, group_df) in enumerate(df.groupby(group_col)):
                color = ENTERPRISE_COLORS.get(group, px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)])
                group_df = group_df.sort_values(x_col)
                
                fig.add_trace(
                    go.Scatter(
                        x=group_df[x_col],
                        y=group_df[secondary_y_col],
                        mode='lines+markers',
                        name=f'{group} - {secondary_y_col}',
                        line=dict(color=color, width=2, dash='dot'),
                        marker=dict(size=6, symbol='square')
                    ),
                    secondary_y=True
                )
        else:
            df_sorted = df.sort_values(x_col)
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[x_col],
                    y=df_sorted[secondary_y_col],
                    mode='lines+markers',
                    name=secondary_y_col,
                    line=dict(color=ENTERPRISE_COLORS['INFO'], width=2, dash='dot'),
                    marker=dict(size=6, symbol='square')
                ),
                secondary_y=True
            )
    
    # Update layout
    layout_updates = {
        'title': {
            'text': title,
            'x': 0.5,
            'font': {'size': 16, 'family': 'Arial'}
        },
        'xaxis': {
            'title': x_col.replace('_', ' ').title(),
            'tickformat': date_format,
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        'yaxis': {
            'title': y_col.replace('_', ' ').title(),
            'showgrid': True,
            'gridcolor': 'lightgray'
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        },
        'hovermode': 'x unified'
    }
    
    if secondary_y_col:
        fig.update_yaxes(title_text=y_col.replace('_', ' ').title(), secondary_y=False)
        fig.update_yaxes(title_text=secondary_y_col.replace('_', ' ').title(), secondary_y=True)
    
    fig.update_layout(**layout_updates)
    
    return fig

def plot_loan_volume_trends(df, date_col='reporting_period', volume_col='loan_count',
                           amount_col='loan_amount_sum', enterprise_col='enterprise_flag',
                           title='Loan Volume and Amount Trends'):
    """
    Specialized time series for loan volume analysis with dual metrics.
    
    Args:
        df (DataFrame): Loan volume data
        date_col (str): Date column
        volume_col (str): Loan count column
        amount_col (str): Loan amount column
        enterprise_col (str): Enterprise grouping column
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Loan volume trend chart
        
    Example:
        fig = plot_loan_volume_trends(monthly_summary, 
                                     title='Monthly Origination Trends')
    """
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[title]
    )
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Plot by enterprise if column exists
    if enterprise_col in df.columns:
        for enterprise in df[enterprise_col].unique():
            if pd.isna(enterprise):
                continue
                
            enterprise_data = df[df[enterprise_col] == enterprise].sort_values(date_col)
            color = ENTERPRISE_COLORS.get(enterprise, ENTERPRISE_COLORS['NEUTRAL'])
            
            # Volume (count) on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=enterprise_data[date_col],
                    y=enterprise_data[volume_col],
                    mode='lines+markers',
                    name=f'{enterprise} Count',
                    line=dict(color=color, width=2),
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
            
            # Amount on secondary y-axis
            if amount_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=enterprise_data[date_col],
                        y=enterprise_data[amount_col] / 1e9,  # Convert to billions
                        mode='lines+markers',
                        name=f'{enterprise} Amount ($B)',
                        line=dict(color=color, width=2, dash='dot'),
                        marker=dict(size=6, symbol='square')
                    ),
                    secondary_y=True
                )
    else:
        # Single series
        df_sorted = df.sort_values(date_col)
        
        fig.add_trace(
            go.Scatter(
                x=df_sorted[date_col],
                y=df_sorted[volume_col],
                mode='lines+markers',
                name='Loan Count',
                line=dict(color=ENTERPRISE_COLORS['NEUTRAL'], width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )
        
        if amount_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[date_col],
                    y=df_sorted[amount_col] / 1e9,
                    mode='lines+markers',
                    name='Amount ($B)',
                    line=dict(color=ENTERPRISE_COLORS['INFO'], width=2, dash='dot'),
                    marker=dict(size=6, symbol='square')
                ),
                secondary_y=True
            )
    
    # Update axes
    fig.update_yaxes(title_text="Loan Count", secondary_y=False)
    fig.update_yaxes(title_text="Amount ($ Billions)", secondary_y=True)
    fig.update_xaxes(title_text="Date")
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        hovermode='x unified'
    )
    
    return fig

# =============================================================================
# GEOGRAPHIC VISUALIZATIONS
# =============================================================================

def plot_choropleth(df, location_col, value_col, location_type='state',
                   title=None, color_scale='Blues', exposure_col=None):
    """
    Create choropleth map normalized by exposure or raw values.
    
    Args:
        df (DataFrame): Data with location and value columns
        location_col (str): Column containing location identifiers
        value_col (str): Column containing values to map
        location_type (str): Type of location ('state', 'county', 'zip')
        title (str): Map title
        color_scale (str): Plotly color scale name
        exposure_col (str): Optional exposure column for normalization
        
    Returns:
        plotly.graph_objects.Figure: Choropleth map
        
    Example:
        fig = plot_choropleth(state_summary, 'state', 'fraud_count',
                             exposure_col='total_loans',
                             title='Fraud Rate by State')
    """
    if title is None:
        title = f'{value_col.replace("_", " ").title()} by {location_type.title()}'
    
    # Calculate rate if exposure column provided
    if exposure_col and exposure_col in df.columns:
        df = df.copy()
        rate_col = f'{value_col}_rate'
        df[rate_col] = (df[value_col] / df[exposure_col] * 10000).round(2)  # Per 10,000
        plot_col = rate_col
        colorbar_title = f'{value_col.replace("_", " ").title()} Rate<br>(per 10,000)'
        hover_template = f'<b>%{{location}}</b><br>{value_col}: %{{customdata[0]}}<br>{exposure_col}: %{{customdata[1]}}<br>Rate: %{{z}}<extra></extra>'
        customdata = df[[value_col, exposure_col]].values
    else:
        plot_col = value_col
        colorbar_title = value_col.replace('_', ' ').title()
        hover_template = f'<b>%{{location}}</b><br>{value_col}: %{{z}}<extra></extra>'
        customdata = None
    
    # Determine location mode and scope
    if location_type == 'state':
        locationmode = 'USA-states'
        scope = 'usa'
        # Ensure state names/codes are properly formatted
        if df[location_col].dtype == 'object':
            # Convert state names to abbreviations if needed
            df = _standardize_state_names(df, location_col)
    else:
        locationmode = 'geojson-id'  # For custom geojson
        scope = 'usa'
    
    # Create choropleth
    fig = go.Figure(data=go.Choropleth(
        locations=df[location_col],
        z=df[plot_col],
        locationmode=locationmode,
        colorscale=color_scale,
        colorbar=dict(title=colorbar_title),
        customdata=customdata,
        hovertemplate=hover_template
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'size': 16, 'family': 'Arial'}
        },
        geo=dict(
            scope=scope,
            projection_type='albers usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        ),
        paper_bgcolor='white'
    )
    
    return fig

def plot_fraud_heatmap(df, state_col='property_state', fraud_col='fraud_count',
                      loan_col='total_loans', title='Fraud Concentration by State'):
    """
    Specialized choropleth for fraud concentration analysis.
    
    Args:
        df (DataFrame): State-level fraud data
        state_col (str): State column
        fraud_col (str): Fraud count column
        loan_col (str): Total loans column for normalization
        title (str): Map title
        
    Returns:
        plotly.graph_objects.Figure: Fraud heatmap
        
    Example:
        fig = plot_fraud_heatmap(state_fraud_summary)
    """
    # Calculate fraud rate per 10,000 loans
    df = df.copy()
    df['fraud_rate'] = (df[fraud_col] / df[loan_col] * 10000).round(2)
    
    # Create choropleth with custom color scale
    fig = go.Figure(data=go.Choropleth(
        locations=df[state_col],
        z=df['fraud_rate'],
        locationmode='USA-states',
        colorscale=[
            [0, '#f7fbff'],      # Light blue for low rates
            [0.2, '#deebf7'],
            [0.4, '#c6dbef'],
            [0.6, '#9ecae1'],
            [0.8, '#6baed6'],
            [1.0, '#d62728']     # Red for high rates
        ],
        colorbar=dict(
            title='Fraud Rate<br>(per 10,000 loans)',
            titleside='right'
        ),
        customdata=df[[fraud_col, loan_col]].values,
        hovertemplate='<b>%{location}</b><br>' +
                     'Fraud Cases: %{customdata[0]}<br>' +
                     'Total Loans: %{customdata[1]:,}<br>' +
                     'Rate: %{z} per 10,000<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'size': 16, 'family': 'Arial'}
        },
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        ),
        paper_bgcolor='white'
    )
    
    return fig

# =============================================================================
# BAR CHARTS AND COMPARISONS
# =============================================================================

def plot_horizontal_bars(df, x_col, y_col, color_col=None, title=None,
                        sort_values=True, top_n=None, show_values=True):
    """
    Create horizontal bar chart with optional grouping and value labels.
    
    Args:
        df (DataFrame): Input data
        x_col (str): X-axis values (numeric)
        y_col (str): Y-axis categories
        color_col (str): Optional color grouping column
        title (str): Chart title
        sort_values (bool): Sort bars by value
        top_n (int): Show only top N categories
        show_values (bool): Display values on bars
        
    Returns:
        plotly.graph_objects.Figure: Horizontal bar chart
        
    Example:
        fig = plot_horizontal_bars(state_summary, 'loan_count', 'state',
                                  title='Loans by State', top_n=10)
    """
    if title is None:
        title = f'{x_col.replace("_", " ").title()} by {y_col.replace("_", " ").title()}'
    
    df_plot = df.copy()
    
    # Sort and filter data
    if sort_values:
        df_plot = df_plot.sort_values(x_col, ascending=True)
    
    if top_n:
        df_plot = df_plot.tail(top_n)
    
    # Create figure
    if color_col and color_col in df.columns:
        # Grouped bar chart
        fig = px.bar(
            df_plot, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            orientation='h',
            title=title,
            color_discrete_map=ENTERPRISE_COLORS
        )
    else:
        # Single series bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=df_plot[x_col],
                y=df_plot[y_col],
                orientation='h',
                marker_color=ENTERPRISE_COLORS['NEUTRAL'],
                text=df_plot[x_col] if show_values else None,
                textposition='outside' if show_values else 'none'
            )
        ])
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'size': 16, 'family': 'Arial'}
        },
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=bool(color_col),
        margin=dict(l=150)  # Extra space for y-axis labels
    )
    
    return fig

def plot_enterprise_comparison(df, metric_cols, enterprise_col='enterprise_flag',
                              title='Enterprise Comparison', chart_type='bar'):
    """
    Create comparison chart between enterprises across multiple metrics.
    
    Args:
        df (DataFrame): Data with enterprise and metric columns
        metric_cols (list): List of metric columns to compare
        enterprise_col (str): Enterprise grouping column
        title (str): Chart title
        chart_type (str): Chart type ('bar', 'radar')
        
    Returns:
        plotly.graph_objects.Figure: Comparison chart
        
    Example:
        metrics = ['loan_count', 'avg_loan_amount', 'fraud_rate']
        fig = plot_enterprise_comparison(summary_df, metrics)
    """
    if chart_type == 'radar':
        return _create_radar_comparison(df, metric_cols, enterprise_col, title)
    else:
        return _create_bar_comparison(df, metric_cols, enterprise_col, title)

def _create_bar_comparison(df, metric_cols, enterprise_col, title):
    """Create grouped bar chart for enterprise comparison."""
    # Reshape data for grouped bar chart
    df_melted = df.melt(
        id_vars=[enterprise_col],
        value_vars=metric_cols,
        var_name='Metric',
        value_name='Value'
    )
    
    fig = px.bar(
        df_melted,
        x='Metric',
        y='Value',
        color=enterprise_col,
        barmode='group',
        title=title,
        color_discrete_map=ENTERPRISE_COLORS
    )
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        xaxis_title='Metrics',
        yaxis_title='Value',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend_title=enterprise_col.replace('_', ' ').title()
    )
    
    return fig

def _create_radar_comparison(df, metric_cols, enterprise_col, title):
    """Create radar chart for enterprise comparison."""
    fig = go.Figure()
    
    # Normalize metrics to 0-100 scale for radar chart
    df_norm = df.copy()
    for col in metric_cols:
        if col in df.columns:
            col_max = df[col].max()
            col_min = df[col].min()
            if col_max > col_min:
                df_norm[col] = ((df[col] - col_min) / (col_max - col_min)) * 100
            else:
                df_norm[col] = 50  # Midpoint if no variation
    
    # Add trace for each enterprise
    for enterprise in df[enterprise_col].unique():
        if pd.isna(enterprise):
            continue
            
        enterprise_data = df_norm[df_norm[enterprise_col] == enterprise]
        if len(enterprise_data) == 0:
            continue
            
        values = [enterprise_data[col].iloc[0] for col in metric_cols]
        values.append(values[0])  # Close the radar chart
        
        metric_labels = [col.replace('_', ' ').title() for col in metric_cols]
        metric_labels.append(metric_labels[0])  # Close the labels
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels,
            fill='toself',
            name=enterprise,
            line_color=ENTERPRISE_COLORS.get(enterprise, ENTERPRISE_COLORS['NEUTRAL'])
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        showlegend=True
    )
    
    return fig

# =============================================================================
# SPECIALIZED ANALYSIS CHARTS
# =============================================================================

def plot_trend_decomposition(df, date_col, value_col, title='Trend Decomposition'):
    """
    Create trend decomposition visualization showing trend, seasonal, and residual components.
    
    Args:
        df (DataFrame): Time series data
        date_col (str): Date column
        value_col (str): Value column to decompose
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Trend decomposition chart
        
    Example:
        fig = plot_trend_decomposition(monthly_data, 'date', 'loan_volume')
    """
    # Simple trend decomposition using moving averages
    df_sorted = df.sort_values(date_col).copy()
    
    # Calculate 12-month moving average as trend
    df_sorted['trend'] = df_sorted[value_col].rolling(window=12, center=True).mean()
    
    # Calculate seasonal component (simplified)
    df_sorted['month'] = pd.to_datetime(df_sorted[date_col]).dt.month
    monthly_avg = df_sorted.groupby('month')[value_col].mean()
    df_sorted['seasonal'] = df_sorted['month'].map(monthly_avg) - df_sorted[value_col].mean()
    
    # Calculate residual
    df_sorted['residual'] = df_sorted[value_col] - df_sorted['trend'] - df_sorted['seasonal']
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
        vertical_spacing=0.08
    )
    
    # Original data
    fig.add_trace(
        go.Scatter(x=df_sorted[date_col], y=df_sorted[value_col], 
                  mode='lines', name='Original', line=dict(color=ENTERPRISE_COLORS['NEUTRAL'])),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=df_sorted[date_col], y=df_sorted['trend'], 
                  mode='lines', name='Trend', line=dict(color=ENTERPRISE_COLORS['ENT1'])),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(x=df_sorted[date_col], y=df_sorted['seasonal'], 
                  mode='lines', name='Seasonal', line=dict(color=ENTERPRISE_COLORS['ENT2'])),
        row=3, col=1
    )
    
    # Residual
    fig.add_trace(
        go.Scatter(x=df_sorted[date_col], y=df_sorted['residual'], 
                  mode='lines', name='Residual', line=dict(color=ENTERPRISE_COLORS['WARNING'])),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        showlegend=False,
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_distribution_comparison(df, value_col, group_col=None, 
                                chart_type='histogram', title=None):
    """
    Create distribution comparison chart (histogram, box plot, or violin plot).
    
    Args:
        df (DataFrame): Input data
        value_col (str): Value column to analyze
        group_col (str): Optional grouping column
        chart_type (str): Chart type ('histogram', 'box', 'violin')
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Distribution comparison chart
        
    Example:
        fig = plot_distribution_comparison(loan_df, 'loan_amount', 
                                         group_col='enterprise_flag',
                                         chart_type='box')
    """
    if title is None:
        title = f'Distribution of {value_col.replace("_", " ").title()}'
    
    if chart_type == 'histogram':
        if group_col:
            fig = px.histogram(
                df, x=value_col, color=group_col, 
                marginal='box', nbins=50,
                title=title,
                color_discrete_map=ENTERPRISE_COLORS
            )
        else:
            fig = px.histogram(
                df, x=value_col, nbins=50,
                title=title,
                color_discrete_sequence=[ENTERPRISE_COLORS['NEUTRAL']]
            )
    
    elif chart_type == 'box':
        if group_col:
            fig = px.box(
                df, x=group_col, y=value_col,
                title=title,
                color=group_col,
                color_discrete_map=ENTERPRISE_COLORS
            )
        else:
            fig = px.box(
                df, y=value_col,
                title=title,
                color_discrete_sequence=[ENTERPRISE_COLORS['NEUTRAL']]
            )
    
    elif chart_type == 'violin':
        if group_col:
            fig = px.violin(
                df, x=group_col, y=value_col,
                title=title,
                color=group_col,
                color_discrete_map=ENTERPRISE_COLORS
            )
        else:
            fig = px.violin(
                df, y=value_col,
                title=title,
                color_discrete_sequence=[ENTERPRISE_COLORS['NEUTRAL']]
            )
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# =============================================================================
# DASHBOARD AND MULTI-PANEL LAYOUTS
# =============================================================================

def create_dashboard_layout(figures_dict, layout='2x2', title='Dashboard'):
    """
    Combine multiple figures into a dashboard layout.
    
    Args:
        figures_dict (dict): Dictionary of figure names and plotly figures
        layout (str): Layout pattern ('2x2', '1x3', '3x1', 'custom')
        title (str): Dashboard title
        
    Returns:
        plotly.graph_objects.Figure: Combined dashboard
        
    Example:
        figures = {
            'Volume Trends': volume_fig,
            'Geographic Distribution': map_fig,
            'Enterprise Comparison': comparison_fig,
            'Quality Metrics': quality_fig
        }
        dashboard = create_dashboard_layout(figures, layout='2x2')
    """
    n_figures = len(figures_dict)
    
    # Determine subplot layout
    if layout == '2x2':
        rows, cols = 2, 2
    elif layout == '1x3':
        rows, cols = 1, 3
    elif layout == '3x1':
        rows, cols = 3, 1
    elif layout == '1x4':
        rows, cols = 1, 4
    elif layout == '4x1':
        rows, cols = 4, 1
    else:
        # Auto-determine layout
        if n_figures <= 2:
            rows, cols = 1, n_figures
        elif n_figures <= 4:
            rows, cols = 2, 2
        elif n_figures <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
    
    # Create subplot titles
    subplot_titles = list(figures_dict.keys())[:rows*cols]
    
    # Create subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add traces from each figure
    for i, (fig_title, source_fig) in enumerate(figures_dict.items()):
        if i >= rows * cols:
            break
            
        row = i // cols + 1
        col = i % cols + 1
        
        # Add all traces from source figure
        for trace in source_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 20}},
        height=600 * rows,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _standardize_state_names(df, state_col):
    """Convert state names to standard abbreviations for choropleth mapping."""
    state_mapping = {
        'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
        'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
        'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
        'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
        'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
        'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
        'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
        'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
        'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
        'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
        'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
        'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
        'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC'
    }
    
    df = df.copy()
    
    # Convert to lowercase for matching
    state_lower = df[state_col].astype(str).str.lower().str.strip()
    
    # Map full names to abbreviations
    df[state_col] = state_lower.map(state_mapping).fillna(df[state_col])
    
    return df

def apply_enterprise_styling(fig, enterprise_col_data=None):
    """
    Apply consistent enterprise color styling to any plotly figure.
    
    Args:
        fig (plotly.graph_objects.Figure): Figure to style
        enterprise_col_data (Series): Optional enterprise data for color mapping
        
    Returns:
        plotly.graph_objects.Figure: Styled figure
        
    Example:
        fig = apply_enterprise_styling(fig, df['enterprise_flag'])
    """
    # Update traces with enterprise colors
    if enterprise_col_data is not None:
        enterprises = enterprise_col_data.unique()
        for i, trace in enumerate(fig.data):
            if i < len(enterprises):
                enterprise = enterprises[i]
                color = ENTERPRISE_COLORS.get(enterprise, ENTERPRISE_COLORS['NEUTRAL'])
                trace.update(marker_color=color)
    
    # Apply consistent layout styling
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial', 'size': 12},
        title={'font': {'size': 16, 'family': 'Arial'}},
        legend={
            'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': 'rgba(0,0,0,0.2)',
            'borderwidth': 1
        }
    )
    
    return fig

def create_summary_table_visual(df, title='Summary Statistics', 
                               format_currency=None, format_percentage=None):
    """
    Create a formatted table visualization from summary statistics.
    
    Args:
        df (DataFrame): Summary statistics DataFrame
        title (str): Table title
        format_currency (list): Columns to format as currency
        format_percentage (list): Columns to format as percentage
        
    Returns:
        plotly.graph_objects.Figure: Table visualization
        
    Example:
        summary_table = create_summary_table_visual(
            stats_df, 
            format_currency=['loan_amount'],
            format_percentage=['fraud_rate']
        )
    """
    if format_currency is None:
        format_currency = []
    if format_percentage is None:
        format_percentage = []
    
    # Format cell values
    df_formatted = df.copy()
    
    for col in format_currency:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f'${x:,.0f}' if pd.notna(x) else '')
    
    for col in format_percentage:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else '')
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_formatted.columns),
            fill_color=ENTERPRISE_COLORS['ENT1'],
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[df_formatted[col] for col in df_formatted.columns],
            fill_color=[['white', '#f0f0f0'] * len(df_formatted)],
            align='center',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# =============================================================================
# TESTING AND SAMPLE DATA GENERATION
# =============================================================================

def generate_sample_data(data_type='loan', n_records=1000):
    """
    Generate sample data for testing visualization functions.
    
    Args:
        data_type (str): Type of data ('loan', 'fraud', 'time_series')
        n_records (int): Number of records to generate
        
    Returns:
        DataFrame: Sample data
        
    Example:
        sample_df = generate_sample_data('loan', 500)
        fig = plot_time_series(sample_df, 'date', 'loan_amount')
    """
    np.random.seed(42)  # For reproducible results
    
    if data_type == 'loan':
        return pd.DataFrame({
            'loan_sequence_number': [f'LSN{i:06d}' for i in range(n_records)],
            'origination_date': pd.date_range('2020-01-01', periods=n_records, freq='D'),
            'loan_amount': np.random.normal(300000, 100000, n_records).clip(50000, 2000000),
            'interest_rate': np.random.normal(4.5, 1.0, n_records).clip(2.0, 8.0),
            'property_state': np.random.choice(['CA', 'TX', 'FL', 'NY', 'PA'], n_records),
            'enterprise_flag': np.random.choice(['ENT1', 'ENT2'], n_records)
        })
    
    elif data_type == 'fraud':
        return pd.DataFrame({
            'bsa_id': [f'BSA{i:08d}' for i in range(n_records)],
            'report_date': pd.date_range('2020-01-01', periods=n_records, freq='W'),
            'fraud_type': np.random.choice(['Occupancy', 'Income', 'Asset', 'Identity'], n_records),
            'amount': np.random.lognormal(12, 1, n_records),
            'property_state': np.random.choice(['CA', 'TX', 'FL', 'NY', 'PA'], n_records),
            'enterprise_flag': np.random.choice(['ENT1', 'ENT2'], n_records)
        })
    
    elif data_type == 'time_series':
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
        trend = np.linspace(1000, 1500, len(dates))
        seasonal = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 50, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'loan_count': (trend + seasonal + noise).astype(int),
            'loan_amount_sum': (trend + seasonal + noise) * 250000,
            'enterprise_flag': np.random.choice(['ENT1', 'ENT2'], len(dates))
        })
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def test_visualization_functions():
    """
    Test all visualization functions with sample data.
    
    Returns:
        dict: Test results with generated figures
        
    Example:
        test_results = test_visualization_functions()
        test_results['time_series'].show()
    """
    print("🧪 Testing visualization functions...")
    
    test_results = {}
    
    try:
        # Generate sample data
        loan_data = generate_sample_data('loan', 500)
        fraud_data = generate_sample_data('fraud', 200)
        time_data = generate_sample_data('time_series', 60)
        
        print("✓ Sample data generated")
        
        # Test time series plot
        time_fig = plot_time_series(
            time_data, 'date', 'loan_count', 
            group_col='enterprise_flag',
            title='Test Time Series'
        )
        test_results['time_series'] = time_fig
        print("✓ Time series plot test passed")
        
        # Test horizontal bar chart
        state_summary = loan_data.groupby('property_state').agg({
            'loan_amount': 'sum',
            'loan_sequence_number': 'count'
        }).reset_index()
        
        bar_fig = plot_horizontal_bars(
            state_summary, 'loan_amount', 'property_state',
            title='Test Bar Chart', top_n=5
        )
        test_results['horizontal_bars'] = bar_fig
        print("✓ Horizontal bar chart test passed")
        
        # Test distribution comparison
        dist_fig = plot_distribution_comparison(
            loan_data, 'loan_amount', 
            group_col='enterprise_flag',
            chart_type='box'
        )
        test_results['distribution'] = dist_fig
        print("✓ Distribution comparison test passed")
        
        # Test choropleth (with mock state data)
        state_fraud = fraud_data.groupby('property_state').agg({
            'bsa_id': 'count',
            'amount': 'sum'
        }).reset_index()
        state_fraud.columns = ['state', 'fraud_count', 'fraud_amount']
        
        # Add mock loan counts for rate calculation
        state_fraud['total_loans'] = np.random.randint(1000, 10000, len(state_fraud))
        
        map_fig = plot_choropleth(
            state_fraud, 'state', 'fraud_count',
            exposure_col='total_loans',
            title='Test Choropleth'
        )
        test_results['choropleth'] = map_fig
        print("✓ Choropleth map test passed")
        
        print("✅ All visualization tests passed!")
        return test_results
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return {'error': str(e)}

# =============================================================================
# MATPLOTLIB STATIC VERSIONS (FOR PDF EXPORT)
# =============================================================================

def create_static_time_series(df, x_col, y_col, group_col=None, 
                             title=None, figsize=(12, 6), save_path=None):
    """
    Create static matplotlib version of time series for PDF export.
    
    Args:
        df (DataFrame): Input data
        x_col (str): Date column
        y_col (str): Value column
        group_col (str): Optional grouping column
        title (str): Chart title
        figsize (tuple): Figure size
        save_path (str): Optional save path
        
    Returns:
        matplotlib.figure.Figure: Static figure
        
    Example:
        fig = create_static_time_series(df, 'date', 'volume', 
                                       save_path='volume_trend.png')
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    if title is None:
        title = f'{y_col.replace("_", " ").title()} Over Time'
    
    # Ensure x column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df[x_col] = pd.to_datetime(df[x_col])
    
    if group_col and group_col in df.columns:
        for i, (group, group_df) in enumerate(df.groupby(group_col)):
            color = ENTERPRISE_COLORS.get(group, f'C{i}')
            group_df = group_df.sort_values(x_col)
            
            ax.plot(
                group_df[x_col], group_df[y_col],
                color=color, linewidth=2, marker='o', markersize=4,
                label=group
            )
    else:
        df_sorted = df.sort_values(x_col)
        ax.plot(
            df_sorted[x_col], df_sorted[y_col],
            color=ENTERPRISE_COLORS['NEUTRAL'], linewidth=2, 
            marker='o', markersize=4
        )
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if group_col:
        ax.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Static figure saved: {save_path}")
    
    return fig

def create_static_bar_chart(df, x_col, y_col, title=None, 
                           figsize=(10, 8), horizontal=True, save_path=None):
    """
    Create static matplotlib bar chart for PDF export.
    
    Args:
        df (DataFrame): Input data
        x_col (str): Value column
        y_col (str): Category column
        title (str): Chart title
        figsize (tuple): Figure size
        horizontal (bool): Horizontal orientation
        save_path (str): Optional save path
        
    Returns:
        matplotlib.figure.Figure: Static figure
        
    Example:
        fig = create_static_bar_chart(state_df, 'loan_count', 'state')
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    if title is None:
        title = f'{x_col.replace("_", " ").title()} by {y_col.replace("_", " ").title()}'
    
    # Sort data
    df_sorted = df.sort_values(x_col, ascending=horizontal)
    
    if horizontal:
        bars = ax.barh(df_sorted[y_col], df_sorted[x_col], 
                      color=ENTERPRISE_COLORS['NEUTRAL'], alpha=0.8)
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    else:
        bars = ax.bar(df_sorted[y_col], df_sorted[x_col], 
                     color=ENTERPRISE_COLORS['NEUTRAL'], alpha=0.8)
        ax.set_ylabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel(y_col.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        if horizontal:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:,.0f}', ha='left', va='center', fontsize=10)
        else:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Static figure saved: {save_path}")
    
    return fig

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_results = test_visualization_functions()
    if 'error' not in test_results:
        print(f"✅ Generated {len(test_results)} test visualizations")
        print("   Use test_results['figure_name'].show() to display figures")
    else:
        print(f"❌ Test failed: {test_results['error']}")