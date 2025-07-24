from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from scipy.stats import skew
import uuid
from bs4 import BeautifulSoup
import requests
import pdfplumber
from PIL import Image
import pytesseract
import io 
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shutil
from scraper import process_image
import re
from typing import Dict, List, Any
# from chatbot import get_chat_response

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.1.11:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#chatbot code 
# api_key = "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"
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ("exit", "quit", "bye"):
#         break
#     print("Bot:", get_chat_response(user_input, api_key))
    
    
UPLOAD_DIR = "uploaded_files"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/uploaded", StaticFiles(directory=UPLOAD_DIR), name="uploaded")


def generate_insights(df: pd.DataFrame) -> List[str]:
    insights = []
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        insights.append(f"⚠️ Dataset contains {total_missing} missing values.")

    numeric_df = df.select_dtypes(include=["number"])
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        col_max = col_data.max()
        col_min = col_data.min()
        col_mean = col_data.mean()
        col_median = col_data.median()
        std_dev = col_data.std()
        col_skew = skew(col_data)

        insights.append(f"📊 '{col}' ranges from {col_min} to {col_max}, avg: {round(col_mean, 2)}, median: {col_median}.")
        if std_dev > col_mean:
            insights.append(f"📉 '{col}' shows high variability (std: {round(std_dev, 2)}).")
        if col_skew > 1:
            insights.append(f"↗️ '{col}' is right-skewed.")
        elif col_skew < -1:
            insights.append(f"↙️ '{col}' is left-skewed.")

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        outliers = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()
        if outliers > 0:
            insights.append(f"🚨 '{col}' has {outliers} potential outliers.")

    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.8:
                        insights.append(f"🔗 '{col1}' and '{col2}' have strong correlation ({round(corr_val, 2)}).")

    cat_df = df.select_dtypes(include=["object", "category"])
    for col in cat_df.columns:
        value_counts = cat_df[col].value_counts()
        if not value_counts.empty:
            top_val = value_counts.index[0]
            freq = value_counts.iloc[0]
            percent = round((freq / len(cat_df)) * 100, 2)
            insights.append(f"🗂️ Most common value in '{col}' is '{top_val}' ({freq} times, {percent}%).")
            if percent > 60:
                insights.append(f"⚠️ '{col}' is highly imbalanced.")

    return insights


def setup_professional_style():
    """Setup professional matplotlib and seaborn styling"""
    # Custom color palette
    professional_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']
    
    # Set matplotlib parameters
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'font.family': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#666666',
        'ytick.color': '#666666'
    })
    
    # Set seaborn style
    sns.set_palette(professional_colors)
    sns.set_style("whitegrid", {
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.3,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    return professional_colors


def add_chart_branding(ax, title, subtitle=None):
    """Add professional branding and titles to charts"""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    if subtitle:
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, ha='center', 
                fontsize=10, style='italic', color='#7F8C8D')


def save_high_quality_plot(fig, filename, output_dir, dpi=300):
    """Save plot with high quality settings"""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, 
                bbox_inches='tight', 
                dpi=dpi, 
                facecolor='white',
                edgecolor='none',
                format='png',
                pad_inches=0.2)
    plt.close(fig)
    return f"/uploaded/{filename}"


def create_enhanced_correlation_heatmap(df, numeric_cols, output_dir):
    """Create professional correlation heatmap"""
    if len(numeric_cols) < 2:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Custom colormap
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abd9e9', '#74add1', '#4575b4']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap=cmap,
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={
                    'shrink': 0.8,
                    'label': 'Correlation Coefficient',
                    'orientation': 'vertical'
                },
                ax=ax,
                linewidths=0.5,
                linecolor='white')
    
    add_chart_branding(ax, 'Feature Correlation Analysis', 
                      'Higher absolute values indicate stronger relationships')
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    filename = f"correlation{uuid.uuid4().hex[:8]}.png"
    return save_high_quality_plot(fig, filename, output_dir)


def create_enhanced_histogram(df, col, output_dir):
    """Create professional histogram with statistical annotations"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = df[col].dropna()
    
    # Create histogram with KDE
    n, bins, patches = ax.hist(data, bins=30, alpha=0.7, color='#3498db', 
                              edgecolor='white', linewidth=1.2)
    
    # Add KDE line
    ax2 = ax.twinx()
    sns.kdeplot(data=data, ax=ax2, color='#e74c3c', linewidth=2.5, alpha=0.8)
    ax2.set_ylabel('Density', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Add statistics
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='#f39c12', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.2f}')
    
    # Add statistics box
    stats_text = f'Count: {len(data):,}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {std_val:.2f}'
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
            fontsize=9)
    
    add_chart_branding(ax, f'Distribution Analysis: {col}',
                      f'Statistical distribution with key metrics')
    
    ax.set_xlabel(col, fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.legend(loc='upper left')
    
    # Clean column name for filename
    clean_col = ''.join(c for c in col if c.isalnum())
    filename = f"hist{clean_col}{uuid.uuid4().hex[:8]}.png"
    return save_high_quality_plot(fig, filename, output_dir)


def create_enhanced_bar_chart(df, col, output_dir):
    """Create professional bar chart with enhanced styling"""
    val_counts = df[col].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create gradient colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(val_counts)))
    
    bars = ax.bar(range(len(val_counts)), val_counts.values, 
                  color=colors, alpha=0.8, edgecolor='white', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, val_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # Customize x-axis
    ax.set_xticks(range(len(val_counts)))
    ax.set_xticklabels([str(x)[:15] + '...' if len(str(x)) > 15 else str(x) 
                       for x in val_counts.index], rotation=45, ha='right')
    
    add_chart_branding(ax, f'Top Categories: {col}',
                      f'Distribution of top {len(val_counts)} categories')
    
    ax.set_xlabel('Categories', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    
    # Add percentage annotations
    total = val_counts.sum()
    for i, (bar, value) in enumerate(zip(bars, val_counts.values)):
        percentage = (value / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{percentage:.1f}%', ha='center', va='center', 
                color='white', fontweight='bold', fontsize=9)
    
    # Clean column name for filename
    clean_col = ''.join(c for c in col if c.isalnum())
    filename = f"bar{clean_col}{uuid.uuid4().hex[:8]}.png"
    return save_high_quality_plot(fig, filename, output_dir)


def create_enhanced_pie_chart(df, col, output_dir):
    """Create professional pie chart with modern styling"""
    val_counts = df[col].value_counts().head(8)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Modern color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    # Create pie chart with modern styling
    wedges, texts, autotexts = ax.pie(val_counts.values, 
                                     labels=val_counts.index,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors[:len(val_counts)],
                                     explode=[0.05] * len(val_counts),  # Separate slices
                                     shadow=True,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    # Enhance text styling
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    add_chart_branding(ax, f'Distribution Breakdown: {col}',
                      'Proportional representation of categories')
    
    # Add legend
    ax.legend(wedges, [f'{label}: {count:,}' for label, count in val_counts.items()],
              title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Clean column name for filename
    clean_col = ''.join(c for c in col if c.isalnum())
    filename = f"pie{clean_col}{uuid.uuid4().hex[:8]}.png"
    return save_high_quality_plot(fig, filename, output_dir)


def create_box_plots(df, numeric_cols, output_dir):
    """Create professional box plots for outlier analysis"""
    if len(numeric_cols) < 1:
        return []
    
    visuals = []
    
    # Individual box plots for each numeric column
    for col in numeric_cols[:4]:  # Limit to first 4 columns
        fig, ax = plt.subplots(figsize=(8, 6))
        
        box_plot = ax.boxplot(df[col].dropna(), patch_artist=True, 
                             boxprops=dict(facecolor='#3498db', alpha=0.7),
                             medianprops=dict(color='#e74c3c', linewidth=2))
        
        # Add scatter points for outliers
        data = df[col].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
        
        if len(outliers) > 0:
            ax.scatter([1] * len(outliers), outliers, 
                      alpha=0.6, color='#e74c3c', s=30, zorder=10)
        
        add_chart_branding(ax, f'Outlier Analysis: {col}',
                          f'Box plot showing distribution and outliers')
        
        ax.set_ylabel(col, fontweight='bold')
        ax.set_xticklabels([col])
        
        # Clean column name for filename
        clean_col = ''.join(c for c in col if c.isalnum())
        filename = f"box{clean_col}{uuid.uuid4().hex[:8]}.png"
        visuals.append(save_high_quality_plot(fig, filename, output_dir))
    
    return visuals


def generate_visualizations(df: pd.DataFrame, output_dir=UPLOAD_DIR):
    """Enhanced visualization generation with professional styling"""
    # Setup professional styling
    professional_colors = setup_professional_style()
    
    visuals = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    try:
        # 1. Enhanced Correlation Heatmap
        if len(numeric_cols) >= 2:
            heatmap = create_enhanced_correlation_heatmap(df, numeric_cols, output_dir)
            if heatmap:
                visuals.append(heatmap)
        
        # 2. Enhanced Histograms (limit to first 3 numeric columns)
        for col in numeric_cols[:3]:
            histogram = create_enhanced_histogram(df, col, output_dir)
            visuals.append(histogram)
        
        # 3. Enhanced Bar Charts for categorical data
        for col in categorical_cols[:2]:
            if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                bar_chart = create_enhanced_bar_chart(df, col, output_dir)
                visuals.append(bar_chart)
        
        # 4. Enhanced Pie Charts for categorical data with few categories
        for col in categorical_cols[:2]:
            if 2 <= df[col].nunique() <= 8:
                pie_chart = create_enhanced_pie_chart(df, col, output_dir)
                visuals.append(pie_chart)
        
        # 5. Box Plots for outlier analysis
        box_plots = create_box_plots(df, numeric_cols, output_dir)
        visuals.extend(box_plots)
        
        return visuals
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return visuals


# def generate_dashboard_title(df: pd.DataFrame) -> dict:
#     """Generate a professional dashboard title based on the data characteristics"""
#     # Get basic dataset info
#     num_rows = df.shape[0]
#     num_cols = df.shape[1]
    
#     # Get column types
#     numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
#     # Determine the main focus of the data
#     if len(numeric_cols) > len(categorical_cols):
#         data_type = "Analytics"
#     else:
#         data_type = "Categorical Analysis"
    
#     # Get time-based columns if they exist
#     time_cols = [col for col in df.columns if any(time_term in col.lower() for time_term in ['date', 'time', 'year', 'month', 'day'])]
    
#     # Get business-related columns
#     business_cols = [col for col in df.columns if any(term in col.lower() for term in 
#                     ['sales', 'revenue', 'profit', 'cost', 'price', 'customer', 'product', 'order'])]
    
#     # Generate title components
#     title_parts = []
    
#     # Add business context if available
#     if business_cols:
#         main_business_col = business_cols[0].replace('_', ' ').title()
#         title_parts.append(f"{main_business_col}")
    
#     # Add time context if available
#     if time_cols:
#         time_col = time_cols[0].replace('_', ' ').title()
#         title_parts.append(f"by {time_col}")
    
#     # Add data type
#     title_parts.append(data_type)
    
#     # Add dataset size context
#     if num_rows > 1000:
#         size_context = "Large-Scale"
#     elif num_rows > 100:
#         size_context = "Comprehensive"
#     else:
#         size_context = "Detailed"
#     title_parts.append(f"{size_context} Dashboard")
    
#     # Combine parts into final title
#     main_title = " | ".join(title_parts)
    
#     # Add subtitle with key metrics
#     subtitle = f"Analyzing {num_rows:,} records across {num_cols} dimensions"
    
#     # Generate title variations for different contexts
#     title_info = {
#         "main_title": main_title,
#         "short_title": title_parts[0] if title_parts else "Data Dashboard",
#         "subtitle": subtitle,
#         "header": {
#             "title": main_title,
#             "subtitle": subtitle,
#             "icon": "📊" if len(numeric_cols) > len(categorical_cols) else "📈"
#         },
#         "breadcrumb": " > ".join(title_parts),
#         "page_title": f"{main_title} - Data Analysis Dashboard",
#         "metrics": {
#             "total_records": num_rows,
#             "total_dimensions": num_cols,
#             "numeric_columns": len(numeric_cols),
#             "categorical_columns": len(categorical_cols),
#             "has_time_data": bool(time_cols),
#             "has_business_metrics": bool(business_cols)
#         },
#         "context": {
#             "data_type": data_type,
#             "size_context": size_context,
#             "has_time_series": bool(time_cols),
#             "has_business_data": bool(business_cols),
#             "primary_metric": business_cols[0] if business_cols else None,
#             "time_dimension": time_cols[0] if time_cols else None
#         }
#     }
    
#     return title_info

def generate_dashboard_title(df: pd.DataFrame, filename: str = None) -> dict:
    """Generate intelligent, context-aware dashboard titles based on data analysis"""
    
    # Get basic dataset info
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    columns = [col.lower().strip() for col in df.columns]
    
    # Industry/Domain detection patterns
    industry_patterns = {
        'retail': ['sales', 'product', 'customer', 'order', 'inventory', 'store', 'purchase', 'revenue'],
        'finance': ['transaction', 'account', 'balance', 'payment', 'investment', 'portfolio', 'credit', 'loan'],
        'hr': ['employee', 'salary', 'department', 'performance', 'hire', 'staff', 'payroll'],
        'marketing': ['campaign', 'leads', 'conversion', 'clicks', 'impressions', 'ctr', 'roi', 'engagement'],
        'healthcare': ['patient', 'diagnosis', 'treatment', 'medical', 'hospital', 'doctor', 'medication'],
        'education': ['student', 'grade', 'course', 'exam', 'teacher', 'school', 'enrollment'],
        'logistics': ['shipment', 'delivery', 'warehouse', 'supplier', 'freight', 'inventory', 'tracking'],
        'technology': ['user', 'session', 'api', 'performance', 'server', 'application', 'database'],
        'sports': ['player', 'team', 'score', 'match', 'season', 'statistics', 'performance', 'league']
    }
    
    # Business metrics patterns
    metric_patterns = {
        'sales': ['sales', 'revenue', 'income', 'earnings', 'profit'],
        'financial': ['cost', 'expense', 'budget', 'price', 'amount', 'value'],
        'performance': ['rating', 'score', 'performance', 'efficiency', 'productivity'],
        'customer': ['customer', 'client', 'user', 'member', 'subscriber'],
        'operational': ['quantity', 'volume', 'count', 'frequency', 'duration'],
        'geographical': ['region', 'country', 'state', 'city', 'location', 'area']
    }
    
    # Time-based patterns
    time_patterns = ['date', 'time', 'year', 'month', 'quarter', 'week', 'day', 'period']
    
    # Extract company/brand name from filename
    def extract_brand_from_filename(filename: str) -> str:
        if not filename:
            return None
        
        # Remove file extension and common prefixes
        name = re.sub(r'\.(csv|xlsx|xls)$', '', filename, flags=re.IGNORECASE)
        name = re.sub(r'^(data_|report_|analysis_|dashboard_)', '', name, flags=re.IGNORECASE)
        
        # Split by common separators and take the first meaningful part
        parts = re.split(r'[-_\s]+', name)
        if parts:
            brand = parts[0].strip()
            # Capitalize properly
            if brand and len(brand) > 1:
                return brand.title()
        return None
    
    # Detect industry based on column names
    def detect_industry(columns: List[str]) -> str:
        industry_scores = {}
        for industry, keywords in industry_patterns.items():
            score = sum(1 for col in columns for keyword in keywords if keyword in col)
            if score > 0:
                industry_scores[industry] = score
        
        if industry_scores:
            return max(industry_scores, key=industry_scores.get)
        return None
    
    # Detect primary business metrics
    def detect_primary_metrics(columns: List[str]) -> List[str]:
        found_metrics = []
        for metric_type, keywords in metric_patterns.items():
            for col in columns:
                for keyword in keywords:
                    if keyword in col:
                        found_metrics.append(metric_type)
                        break
                if metric_type in found_metrics:
                    break
        return list(set(found_metrics))
    
    # Detect time dimensions
    def detect_time_dimension(columns: List[str]) -> str:
        for col in columns:
            for time_term in time_patterns:
                if time_term in col:
                    return time_term.title()
        return None
    
    # Get actual data insights for smarter titles
    def get_data_insights(df: pd.DataFrame) -> Dict[str, Any]:
        insights = {}
        
        # Check for categorical columns with specific patterns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            col_lower = col.lower()
            unique_values = df[col].dropna().unique()
            
            # Check if it's a product/brand column
            if any(term in col_lower for term in ['product', 'brand', 'item', 'model']):
                insights['has_products'] = True
                insights['product_column'] = col
                insights['product_count'] = len(unique_values)
            
            # Check if it's a category column
            elif any(term in col_lower for term in ['category', 'type', 'class', 'segment']):
                insights['has_categories'] = True
                insights['category_column'] = col
                insights['category_count'] = len(unique_values)
            
            # Check if it's a geographical column
            elif any(term in col_lower for term in ['country', 'region', 'state', 'city', 'location']):
                insights['has_geography'] = True
                insights['geo_column'] = col
        
        return insights
    
    # Main title generation logic
    brand_name = extract_brand_from_filename(filename) if filename else None
    industry = detect_industry(columns)
    primary_metrics = detect_primary_metrics(columns)
    time_dimension = detect_time_dimension(columns)
    data_insights = get_data_insights(df)
    
    # Build title components
    title_parts = []
    
    # Add brand/company name if detected
    if brand_name:
        title_parts.append(brand_name)
    
    # Add primary business focus
    if 'sales' in primary_metrics or any('sales' in col for col in columns):
        title_parts.append("Sales Dashboard")
    elif 'financial' in primary_metrics and 'performance' in primary_metrics:
        title_parts.append("Financial Performance Dashboard")
    elif 'customer' in primary_metrics:
        title_parts.append("Customer Analytics Dashboard")
    elif industry == 'retail' and data_insights.get('has_products'):
        title_parts.append("Product Performance Dashboard")
    elif industry == 'marketing':
        title_parts.append("Marketing Analytics Dashboard")
    elif industry == 'hr':
        title_parts.append("HR Analytics Dashboard")
    elif industry == 'finance':
        title_parts.append("Financial Analytics Dashboard")
    elif industry and primary_metrics:
        title_parts.append(f"{industry.title()} {primary_metrics[0].title()} Dashboard")
    elif industry:
        title_parts.append(f"{industry.title()} Analytics Dashboard")
    elif primary_metrics:
        title_parts.append(f"{primary_metrics[0].title()} Analytics Dashboard")
    else:
        # Fallback to column-based title
        key_columns = [col for col in df.columns[:3] if not any(time_term in col.lower() for time_term in time_patterns)]
        if key_columns:
            main_focus = key_columns[0].replace('_', ' ').title()
            title_parts.append(f"{main_focus} Analytics Dashboard")
        else:
            title_parts.append("Business Intelligence Dashboard")
    
    # Generate main title
    if len(title_parts) == 1:
        main_title = title_parts[0]
    else:
        main_title = " - ".join(title_parts)
    
    # Generate contextual subtitle
    subtitle_parts = []
    
    if data_insights.get('has_products'):
        subtitle_parts.append(f"{data_insights['product_count']} products")
    if data_insights.get('has_categories'):
        subtitle_parts.append(f"{data_insights['category_count']} categories")
    if data_insights.get('has_geography'):
        subtitle_parts.append("multi-region analysis")
    if time_dimension:
        subtitle_parts.append(f"tracked over {time_dimension.lower()}")
    
    # Add data scale context
    if num_rows > 10000:
        subtitle_parts.append(f"{num_rows:,} records")
    elif num_rows > 1000:
        subtitle_parts.append(f"{num_rows:,} data points")
    
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else f"Comprehensive analysis of {num_rows:,} records"
    
    # Generate short title for navigation
    if brand_name:
        short_title = f"{brand_name} Dashboard"
    elif industry:
        short_title = f"{industry.title()} Dashboard"
    else:
        short_title = "Analytics Dashboard"
    
    # Return comprehensive title information
    return {
        "main_title": main_title,
        "short_title": short_title,
        "subtitle": subtitle,
        "header": {
            "title": main_title,
            "subtitle": subtitle,
            "icon": "📊" if 'sales' in primary_metrics else "📈"
        },
        "breadcrumb": " > ".join([short_title, "Analysis"]),
        "page_title": f"{main_title} - Business Intelligence",
        "metrics": {
            "total_records": num_rows,
            "total_dimensions": num_cols,
            "detected_industry": industry,
            "primary_metrics": primary_metrics,
            "has_time_data": bool(time_dimension),
            "brand_detected": bool(brand_name)
        },
        "context": {
            "industry": industry,
            "brand": brand_name,
            "primary_focus": primary_metrics[0] if primary_metrics else None,
            "time_dimension": time_dimension,
            "data_insights": data_insights,
            "title_confidence": "high" if (brand_name or industry) else "medium"
        }
    }

# Example usage and testing
def test_title_generation():
    """Test the title generation with sample data"""
    
    # Test case 1: Nike sales data
    nike_data = pd.DataFrame({
        'Product_Name': ['Air Max', 'Air Force', 'Dunk'],
        'Sales_Amount': [1000, 1500, 800],
        'Month': ['Jan', 'Feb', 'Mar'],
        'Region': ['US', 'EU', 'Asia']
    })
    
    nike_title = generate_dashboard_title(nike_data, "nike_sales_2024.csv")
    print("Nike Example:")
    print(f"Main Title: {nike_title['main_title']}")
    print(f"Subtitle: {nike_title['subtitle']}")
    print()
    
    # Test case 2: Financial data
    financial_data = pd.DataFrame({
        'Account_Type': ['Checking', 'Savings', 'Credit'],
        'Balance': [5000, 15000, -2000],
        'Transaction_Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Customer_ID': [1, 2, 3]
    })
    
    financial_title = generate_dashboard_title(financial_data, "bank_transactions.xlsx")
    print("Financial Example:")
    print(f"Main Title: {financial_title['main_title']}")
    print(f"Subtitle: {financial_title['subtitle']}")
    print()
    
    # Test case 3: Generic business data
    generic_data = pd.DataFrame({
        'Revenue': [10000, 12000, 11000],
        'Customers': [100, 120, 110],
        'Quarter': ['Q1', 'Q2', 'Q3']
    })
    
    generic_title = generate_dashboard_title(generic_data, "business_metrics.csv")
    print("Generic Example:")
    print(f"Main Title: {generic_title['main_title']}")
    print(f"Subtitle: {generic_title['subtitle']}")

# Run test
if __name__ == "__main__":
    test_title_generation()


def prepare_visualization_data(df: pd.DataFrame) -> dict:
    """Prepare data for frontend visualizations"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    viz_data = {
        "correlation": None,
        "histograms": [],
        "bar_charts": [],
        "pie_charts": [],
        "box_plots": [],
        "time_series": []
    }
    
    # 1. Correlation Matrix
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        viz_data["correlation"] = {
            "columns": numeric_cols,
            "data": corr_matrix.values.tolist(),
            "type": "heatmap"
        }
    
    # 2. Histograms for numeric columns
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        data = df[col].dropna()
        hist, bins = np.histogram(data, bins=30)
        viz_data["histograms"].append({
            "column": col,
            "type": "histogram",
            "data": {
                "values": hist.tolist(),
                "bins": bins.tolist(),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max())
            }
        })
    
    # 3. Bar Charts for categorical columns
    for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
        if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
            value_counts = df[col].value_counts().head(10)
            viz_data["bar_charts"].append({
                "column": col,
                "type": "bar",
                "data": {
                    "labels": value_counts.index.tolist(),
                    "values": value_counts.values.tolist(),
                    "percentages": (value_counts.values / len(df) * 100).tolist()
                }
            })
    
    # 4. Pie Charts for categorical columns with few categories
    for col in categorical_cols[:2]:
        if 2 <= df[col].nunique() <= 8:
            value_counts = df[col].value_counts()
            viz_data["pie_charts"].append({
                "column": col,
                "type": "pie",
                "data": {
                    "labels": value_counts.index.tolist(),
                    "values": value_counts.values.tolist(),
                    "percentages": (value_counts.values / len(df) * 100).tolist()
                }
            })
    
    # 5. Box Plots for numeric columns
    for col in numeric_cols[:4]:  # Limit to first 4 numeric columns
        data = df[col].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
        
        viz_data["box_plots"].append({
            "column": col,
            "type": "box",
            "data": {
                "min": float(data.min()),
                "q1": float(q1),
                "median": float(data.median()),
                "q3": float(q3),
                "max": float(data.max()),
                "outliers": outliers.tolist(),
                "mean": float(data.mean())
            }
        })
    
    # 6. Time Series (if time columns exist)
    time_cols = [col for col in df.columns if any(time_term in col.lower() 
                for time_term in ['date', 'time', 'year', 'month', 'day'])]
    
    if time_cols and numeric_cols:
        time_col = time_cols[0]
        numeric_col = numeric_cols[0]  # Use first numeric column for time series
        
        # Sort by time and group if needed
        if df[time_col].dtype == 'datetime64[ns]':
            time_series = df.sort_values(time_col).groupby(time_col)[numeric_col].mean()
            viz_data["time_series"].append({
                "time_column": time_col,
                "value_column": numeric_col,
                "type": "line",
                "data": {
                    "times": time_series.index.strftime('%Y-%m-%d').tolist(),
                    "values": time_series.values.tolist()
                }
            })
    
    return viz_data

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_location)
        else:
            df = pd.read_excel(file_location)

        df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
        preview = df_cleaned.head().to_dict(orient="records")

        # Generate dashboard title and metadata
        dashboard_info = generate_dashboard_title(df_cleaned)

        # Prepare visualization data
        visualization_data = prepare_visualization_data(df_cleaned)

        # Basic summary
        summary = {
            "rows": df_cleaned.shape[0],
            "columns": df_cleaned.shape[1],
            "columns_list": df_cleaned.columns.tolist(),
            "missing_values": int(df_cleaned.isnull().sum().sum())
        }

        # Process numeric data for statistics
        numeric_df = df_cleaned.select_dtypes(include=["number"])
        stats = {}
        if not numeric_df.empty:
            for col in numeric_df.columns:
                col_stats = {
                    "sum": float(numeric_df[col].sum()),
                    "mean": float(numeric_df[col].mean()),
                    "min": float(numeric_df[col].min()),
                    "max": float(numeric_df[col].max()),
                    "median": float(numeric_df[col].median()),
                    "std_dev": float(numeric_df[col].std())
                }
                stats[col] = col_stats

        # Generate insights
        insights = generate_insights(df_cleaned)

        # 🔍 Generate Axis Recommendations using ColumnAnalyzer
        axis_recommendations = analyzer.generate_recommendations(df_cleaned)

        # Construct final response
        response_data = {
            "status": "success",
            "message": "✅ File processed successfully",
            "filename": file.filename,
            "path": f"/uploaded/{file.filename}",
            "dashboard": dashboard_info,
            "summary": summary,
            "preview": preview,
            "statistics": stats,
            "insights": insights,
            "visualizations": visualization_data,
            "axis_recommendations": axis_recommendations.dict(),  # ⭐ Add this line
            "raw_data": {
                "columns": df_cleaned.columns.tolist(),
                "data": df_cleaned.head(100).to_dict(orient="records")
            }
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error processing file: {str(e)}"
            }
        )


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process an image file
    """
    try:
        # Validate image type
        valid_types = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
        if file.content_type not in valid_types:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Only image files are allowed (PNG, JPG, JPEG, WEBP)."
                }
            )

        # Generate a unique filename
        unique_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Process the image using OCR
            output_path = process_image(file_path, 'excel')
            
            if not output_path:
                raise Exception("Failed to generate output file")
            
            # Read the generated Excel file
            df = pd.read_excel(output_path)
            
            # Clean the data
            df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
            preview = df_cleaned.head().to_dict(orient="records")

            # Generate dashboard title and metadata
            dashboard_info = generate_dashboard_title(df_cleaned)

            # Prepare visualization data
            visualization_data = prepare_visualization_data(df_cleaned)

            # Generate summary statistics
            summary = {
                "rows": df_cleaned.shape[0],
                "columns": df_cleaned.shape[1],
                "columns_list": df_cleaned.columns.tolist(),
                "missing_values": int(df_cleaned.isnull().sum().sum())
            }

            # Calculate numeric statistics
            numeric_df = df_cleaned.select_dtypes(include=["number"])
            if numeric_df.empty or numeric_df.shape[1] == 0:
                stats = "No numeric fields in data"
            else:
                stats = {
                    col: {
                        "sum": numeric_df[col].sum().item(),
                        "mean": numeric_df[col].mean().item(),
                        "min": numeric_df[col].min().item(),
                        "max": numeric_df[col].max().item(),
                        "median": numeric_df[col].median().item(),
                        "std_dev": numeric_df[col].std().item()
                    } for col in numeric_df.columns
                }

            # Generate insights
            insights = generate_insights(df_cleaned)

            # Update the response to include visualization data instead of images
            return {
                "status": "success",
                "message": "✅ Image processed successfully",
                "filename": os.path.basename(output_path),
                "path": f"/uploaded/{os.path.basename(output_path)}",
                "dashboard": dashboard_info,
                "summary": summary,
                "preview": preview,
                "statistics": stats,
                "insights": insights,
                "visualizations": visualization_data  # Now contains data for frontend charts
            }
            
        except Exception as e:
            # Clean up the uploaded file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        error_message = str(e)
        if "GOOGLE_API_KEY" in error_message:
            error_message = "Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
        elif "No text was extracted" in error_message:
            error_message = "No text could be extracted from the image. Please ensure the image contains clear, readable text."
        elif "Failed to parse" in error_message:
            error_message = "Failed to parse the extracted text. The image might not contain structured data."
        
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": error_message
            }
        )

# new model code 
class ColumnRecommendation(BaseModel):
    column_name: str
    axis_type: str  # 'x_axis', 'y_axis', 'category', 'filter'
    confidence_score: float
    data_type: str
    reasoning: str
    sample_values: List[Any]

class AxisRecommendations(BaseModel):
    x_axis_candidates: List[ColumnRecommendation]
    y_axis_candidates: List[ColumnRecommendation]
    category_candidates: List[ColumnRecommendation]
    filter_candidates: List[ColumnRecommendation]
    best_combinations: List[Dict[str, Any]]

class ColumnAnalyzer:
    def __init__(self):
        self.x_axis_keywords = [
            'date', 'time', 'month', 'year', 'day', 'period',
            'category', 'type', 'class', 'group', 'segment',
            'product', 'item', 'name', 'region', 'location',
            'department', 'division', 'branch', 'store'
        ]
        
        self.y_axis_keywords = [
            'amount', 'value', 'price', 'cost', 'revenue', 'sales',
            'profit', 'income', 'expense', 'total', 'sum',
            'count', 'quantity', 'number', 'volume', 'rate',
            'percentage', 'ratio', 'score', 'rating', 'index'
        ]
        
        self.category_keywords = [
            'category', 'type', 'class', 'group', 'kind',
            'status', 'state', 'level', 'grade', 'tier'
        ]
        
        # Initialize the ML model
        self.model = self._create_model()
        
    def _create_model(self):
        """Create and train a simple model for axis detection"""
        # This is a simplified model - in production, you'd train on more data
        return {
            'x_axis_model': RandomForestClassifier(n_estimators=50, random_state=42),
            'y_axis_model': RandomForestClassifier(n_estimators=50, random_state=42),
            'scaler': StandardScaler()
        }
    
    def extract_column_features(self, df: pd.DataFrame, column_name: str) -> Dict[str, float]:
        """Extract features from a column for ML model"""
        col_data = df[column_name].dropna()
        
        if len(col_data) == 0:
            return {}
        
        features = {}
        
        # Basic statistics
        features['unique_count'] = col_data.nunique()
        features['total_count'] = len(col_data)
        features['uniqueness_ratio'] = features['unique_count'] / features['total_count']
        features['null_percentage'] = df[column_name].isnull().sum() / len(df)
        
        # Data type features
        features['is_numeric'] = pd.api.types.is_numeric_dtype(col_data)
        features['is_datetime'] = pd.api.types.is_datetime64_any_dtype(col_data)
        features['is_categorical'] = pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object'
        
        # Name-based features
        col_lower = column_name.lower()
        features['has_x_keywords'] = sum(1 for keyword in self.x_axis_keywords if keyword in col_lower)
        features['has_y_keywords'] = sum(1 for keyword in self.y_axis_keywords if keyword in col_lower)
        features['has_category_keywords'] = sum(1 for keyword in self.category_keywords if keyword in col_lower)
        
        # Advanced features for numeric columns
        if features['is_numeric']:
            features['min_value'] = col_data.min()
            features['max_value'] = col_data.max()
            features['mean_value'] = col_data.mean()
            features['std_value'] = col_data.std()
            features['has_negative'] = (col_data < 0).any()
            features['has_decimals'] = (col_data % 1 != 0).any()
            features['value_range'] = features['max_value'] - features['min_value']
            features['coefficient_of_variation'] = features['std_value'] / features['mean_value'] if features['mean_value'] != 0 else 0
        else:
            # Default values for non-numeric columns
            for key in ['min_value', 'max_value', 'mean_value', 'std_value', 'value_range', 'coefficient_of_variation']:
                features[key] = 0
            features['has_negative'] = False
            features['has_decimals'] = False
        
        # Categorical features
        if features['is_categorical']:
            features['avg_string_length'] = col_data.astype(str).str.len().mean()
            features['has_mixed_case'] = col_data.astype(str).str.contains(r'[a-z].*[A-Z]|[A-Z].*[a-z]').any()
        else:
            features['avg_string_length'] = 0
            features['has_mixed_case'] = False
        
        # Cardinality features
        features['is_high_cardinality'] = features['unique_count'] > 50
        features['is_low_cardinality'] = features['unique_count'] < 10
        features['is_binary'] = features['unique_count'] == 2
        
        return features
    
    def predict_axis_suitability(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Predict axis suitability for each column using rule-based approach"""
        predictions = {}
        
        for column in df.columns:
            features = self.extract_column_features(df, column)
            if not features:
                continue
                
            # Rule-based scoring
            x_axis_score = self._calculate_x_axis_score(features, column)
            y_axis_score = self._calculate_y_axis_score(features, column)
            category_score = self._calculate_category_score(features, column)
            filter_score = self._calculate_filter_score(features, column)
            
            predictions[column] = {
                'x_axis': x_axis_score,
                'y_axis': y_axis_score,
                'category': category_score,
                'filter': filter_score
            }
        
        return predictions
    
    def _calculate_x_axis_score(self, features: Dict[str, float], column_name: str) -> float:
        """Calculate X-axis suitability score"""
        score = 0.0
        
        # Keyword matching
        score += features.get('has_x_keywords', 0) * 0.3
        
        # Categorical columns are good for X-axis
        if features.get('is_categorical', False):
            score += 0.4
            
        # Low to medium cardinality is preferred
        if features.get('uniqueness_ratio', 0) < 0.5:
            score += 0.3
        
        # Datetime columns are excellent for X-axis
        if features.get('is_datetime', False):
            score += 0.5
            
        # Penalize high cardinality
        if features.get('is_high_cardinality', False):
            score -= 0.2
            
        # Binary columns can be good for X-axis
        if features.get('is_binary', False):
            score += 0.2
            
        return min(1.0, max(0.0, score))
    
    def _calculate_y_axis_score(self, features: Dict[str, float], column_name: str) -> float:
        """Calculate Y-axis suitability score"""
        score = 0.0
        
        # Keyword matching
        score += features.get('has_y_keywords', 0) * 0.4
        
        # Numeric columns are preferred for Y-axis
        if features.get('is_numeric', False):
            score += 0.5
            
        # Continuous data is better
        if features.get('uniqueness_ratio', 0) > 0.1:
            score += 0.3
            
        # Higher variance indicates measurable data
        if features.get('coefficient_of_variation', 0) > 0.1:
            score += 0.2
            
        # Penalize categorical data
        if features.get('is_categorical', False):
            score -= 0.3
            
        return min(1.0, max(0.0, score))
    
    def _calculate_category_score(self, features: Dict[str, float], column_name: str) -> float:
        """Calculate category/grouping suitability score"""
        score = 0.0
        
        # Keyword matching
        score += features.get('has_category_keywords', 0) * 0.4
        
        # Categorical columns are good for grouping
        if features.get('is_categorical', False):
            score += 0.4
            
        # Low to medium cardinality
        if 2 <= features.get('unique_count', 0) <= 20:
            score += 0.3
            
        # Penalize high cardinality
        if features.get('is_high_cardinality', False):
            score -= 0.4
            
        return min(1.0, max(0.0, score))
    
    def _calculate_filter_score(self, features: Dict[str, float], column_name: str) -> float:
        """Calculate filter suitability score"""
        score = 0.0
        
        # Categorical columns with reasonable cardinality
        if features.get('is_categorical', False) and 2 <= features.get('unique_count', 0) <= 50:
            score += 0.4
            
        # Numeric columns with reasonable range
        if features.get('is_numeric', False) and features.get('unique_count', 0) < 100:
            score += 0.3
            
        # Datetime columns are good for filtering
        if features.get('is_datetime', False):
            score += 0.4
            
        return min(1.0, max(0.0, score))
    
    def generate_recommendations(self, df: pd.DataFrame) -> AxisRecommendations:
        """Generate comprehensive axis recommendations"""
        predictions = self.predict_axis_suitability(df)
        
        # Create recommendations for each axis type
        x_axis_candidates = []
        y_axis_candidates = []
        category_candidates = []
        filter_candidates = []
        
        for column, scores in predictions.items():
            sample_values = df[column].dropna().head(5).tolist()
            
            # X-axis candidates
            if scores['x_axis'] > 0.3:
                x_axis_candidates.append(ColumnRecommendation(
                    column_name=column,
                    axis_type='x_axis',
                    confidence_score=scores['x_axis'],
                    data_type=str(df[column].dtype),
                    reasoning=self._get_reasoning(column, 'x_axis', scores['x_axis']),
                    sample_values=sample_values
                ))
            
            # Y-axis candidates
            if scores['y_axis'] > 0.3:
                y_axis_candidates.append(ColumnRecommendation(
                    column_name=column,
                    axis_type='y_axis',
                    confidence_score=scores['y_axis'],
                    data_type=str(df[column].dtype),
                    reasoning=self._get_reasoning(column, 'y_axis', scores['y_axis']),
                    sample_values=sample_values
                ))
            
            # Category candidates
            if scores['category'] > 0.3:
                category_candidates.append(ColumnRecommendation(
                    column_name=column,
                    axis_type='category',
                    confidence_score=scores['category'],
                    data_type=str(df[column].dtype),
                    reasoning=self._get_reasoning(column, 'category', scores['category']),
                    sample_values=sample_values
                ))
            
            # Filter candidates
            if scores['filter'] > 0.3:
                filter_candidates.append(ColumnRecommendation(
                    column_name=column,
                    axis_type='filter',
                    confidence_score=scores['filter'],
                    data_type=str(df[column].dtype),
                    reasoning=self._get_reasoning(column, 'filter', scores['filter']),
                    sample_values=sample_values
                ))
        
        # Sort by confidence score
        x_axis_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        y_axis_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        category_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        filter_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Generate best combinations
        best_combinations = self._generate_best_combinations(
            x_axis_candidates[:5], y_axis_candidates[:5], category_candidates[:3]
        )
        
        return AxisRecommendations(
            x_axis_candidates=x_axis_candidates,
            y_axis_candidates=y_axis_candidates,
            category_candidates=category_candidates,
            filter_candidates=filter_candidates,
            best_combinations=best_combinations
        )
    
    def _get_reasoning(self, column_name: str, axis_type: str, score: float) -> str:
        """Generate reasoning for the recommendation"""
        col_lower = column_name.lower()
        
        if axis_type == 'x_axis':
            if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year']):
                return f"Time-based column ideal for trend analysis"
            elif any(keyword in col_lower for keyword in ['category', 'type', 'class']):
                return f"Categorical data perfect for grouping on X-axis"
            else:
                return f"Suitable for X-axis with {score:.1%} confidence"
        
        elif axis_type == 'y_axis':
            if any(keyword in col_lower for keyword in ['amount', 'value', 'price', 'profit']):
                return f"Monetary value ideal for Y-axis measurement"
            elif any(keyword in col_lower for keyword in ['count', 'quantity', 'number']):
                return f"Quantitative data perfect for Y-axis"
            else:
                return f"Numeric data suitable for Y-axis with {score:.1%} confidence"
        
        elif axis_type == 'category':
            return f"Good for data grouping and segmentation"
        
        else:  # filter
            return f"Suitable for filtering and data exploration"
    
    def _generate_best_combinations(self, x_candidates: List[ColumnRecommendation], 
                                  y_candidates: List[ColumnRecommendation],
                                  category_candidates: List[ColumnRecommendation]) -> List[Dict[str, Any]]:
        """Generate best X-Y axis combinations"""
        combinations = []
        
        # Top combinations
        for i, x_col in enumerate(x_candidates[:3]):
            for j, y_col in enumerate(y_candidates[:3]):
                if x_col.column_name != y_col.column_name:
                    combined_score = (x_col.confidence_score + y_col.confidence_score) / 2
                    
                    combination = {
                        'x_axis': x_col.column_name,
                        'y_axis': y_col.column_name,
                        'chart_type': self._suggest_chart_type(x_col, y_col),
                        'confidence_score': combined_score,
                        'category': category_candidates[0].column_name if category_candidates else None,
                        'title': f"{y_col.column_name} by {x_col.column_name}",
                        'description': f"Analyze {y_col.column_name} across different {x_col.column_name}"
                    }
                    
                    combinations.append(combination)
        
        # Sort by combined score
        combinations.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return combinations[:5]  # Return top 5 combinations
    
    def _suggest_chart_type(self, x_col: ColumnRecommendation, y_col: ColumnRecommendation) -> str:
        """Suggest the best chart type for the combination"""
        if 'date' in x_col.column_name.lower() or 'time' in x_col.column_name.lower():
            return 'line'
        elif x_col.data_type == 'object' and y_col.data_type in ['int64', 'float64']:
            return 'bar'
        elif x_col.data_type in ['int64', 'float64'] and y_col.data_type in ['int64', 'float64']:
            return 'scatter'
        else:
            return 'bar'

# Global analyzer instance
analyzer = ColumnAnalyzer()

@app.post("/detect-axis-columns", response_model=AxisRecommendations)
async def detect_axis_columns(file: UploadFile = File(...)):
    """Detect best columns for X and Y axes"""
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be CSV or Excel")
    
    try:
        contents = await file.read()
        
        # Read file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Generate recommendations
        recommendations = analyzer.generate_recommendations(df)
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-best-combination")
async def predict_best_combination(file: UploadFile = File(...)):
    """Get the single best X-Y axis combination"""
    
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        recommendations = analyzer.generate_recommendations(df)
        
        if recommendations.best_combinations:
            best = recommendations.best_combinations[0]
            return {
                "status": "success",
                "best_combination": best,
                "alternatives": recommendations.best_combinations[1:3]
            }
        else:
            raise HTTPException(status_code=400, detail="No suitable combinations found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/column-features/{column_name}")
async def get_column_features(column_name: str, file: UploadFile = File(...)):
    """Get detailed features for a specific column"""
    
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        if column_name not in df.columns:
            raise HTTPException(status_code=404, detail="Column not found")
        
        features = analyzer.extract_column_features(df, column_name)
        
        return {
            "column_name": column_name,
            "features": features,
            "sample_data": df[column_name].head(10).tolist(),
            "statistics": {
                "count": len(df[column_name]),
                "unique": df[column_name].nunique(),
                "null_count": df[column_name].isnull().sum()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# --------------- Web Scraping -------------------

# def scrape_table_from_html(url: str):
#     res = requests.get(url)
#     soup = BeautifulSoup(res.content, "html.parser")
#     table = soup.find("table")
#     if not table:
#         raise ValueError("No HTML table found on page.")
#     rows = table.find_all("tr")
#     data = [[col.get_text(strip=True) for col in row.find_all(["td", "th"])] for row in rows]
#     df = pd.DataFrame(data[1:], columns=data[0])
#     return df

# def scrape_pdf_table(url: str):
#     res = requests.get(url)
#     with open("temp_scrape.pdf", "wb") as f:
#         f.write(res.content)
#     with pdfplumber.open("temp_scrape.pdf") as pdf:
#         tables = [p.extract_table() for p in pdf.pages if p.extract_table()]
#     os.remove("temp_scrape.pdf")
#     if not tables:
#         raise ValueError("No table found in PDF.")
#     flat = [row for table in tables for row in table]
#     df = pd.DataFrame(flat[1:], columns=flat[0])
#     return df

# def scrape_image_ocr(url: str):
#     res = requests.get(url)
#     img = Image.open(io.BytesIO(res.content))
#     text = pytesseract.image_to_string(img)
#     lines = [line.split() for line in text.strip().split("\n") if line]
#     if len(lines) < 2:
#         raise ValueError("OCR did not extract structured data.")
#     df = pd.DataFrame(lines[1:], columns=lines[0])
#     return df

# @app.get("/scrape")
# def scrape_url_data(url: str = Query(...)):
#     try:
#         if url.endswith(".pdf"):
#             df = scrape_pdf_table(url)
#         elif url.endswith((".png", ".jpg", ".jpeg")):
#             df = scrape_image_ocr(url)
#         else:
#             df = scrape_table_from_html(url)

#         df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
#         preview = df_cleaned.head().to_dict(orient="records")

#         summary = {
#             "rows": df_cleaned.shape[0],
#             "columns": df_cleaned.shape[1],
#             "columns_list": df_cleaned.columns.tolist(),
#             "missing_values": int(df_cleaned.isnull().sum().sum())
#         }

#         numeric_df = df_cleaned.select_dtypes(include=["number"])
#         stats = {
#             col: {
#                 "sum": numeric_df[col].sum().item(),
#                 "mean": numeric_df[col].mean().item(),
#                 "min": numeric_df[col].min().item(),
#                 "max": numeric_df[col].max().item(),
#                 "median": numeric_df[col].median().item(),
#                 "std_dev": numeric_df[col].std().item()
#             } for col in numeric_df.columns
#         }

#         insights = generate_insights(df_cleaned)
#         visualizations = generate_visualizations(df_cleaned)

#         return {
#             "source": url,
#             "summary": summary,
#             "preview": preview,
#             "statistics": stats,
#             "insights": insights,
#             "visualizations": visualizations
#         }

#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/process-image")
# async def process_image_endpoint(
#     file: UploadFile = File(...),
#     output_format: str = Query("excel", enum=["excel", "csv"])
# ):
#     """
#     Process an uploaded image to extract text and convert to Excel/CSV format
#     """
#     try:
#         # Save the uploaded file
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
#         with open(file_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)
        
#         # Process the image using OCR
#         output_path = process_image(file_path, output_format)
        
#         if output_path:
#             # Read the generated Excel/CSV file
#             if output_format == 'excel':
#                 df = pd.read_excel(output_path)
#             else:
#                 df = pd.read_csv(output_path)
            
#             # Clean the data (same as your existing workflow)
#             df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
#             preview = df_cleaned.head().to_dict(orient="records")

#             # Generate summary statistics
#             summary = {
#                 "rows": df_cleaned.shape[0],
#                 "columns": df_cleaned.shape[1],
#                 "columns_list": df_cleaned.columns.tolist(),
#                 "missing_values": int(df_cleaned.isnull().sum().sum())
#             }

#             # Calculate numeric statistics
#             numeric_df = df_cleaned.select_dtypes(include=["number"])
#             if numeric_df.empty or numeric_df.shape[1] == 0:
#                 stats = "No numeric fields in data"
#             else:
#                 stats = {
#                     col: {
#                         "sum": numeric_df[col].sum().item(),
#                         "mean": numeric_df[col].mean().item(),
#                         "min": numeric_df[col].min().item(),
#                         "max": numeric_df[col].max().item(),
#                         "median": numeric_df[col].median().item(),
#                         "std_dev": numeric_df[col].std().item()
#                     } for col in numeric_df.columns
#                 }

#             # Generate insights and visualizations
#             insights = generate_insights(df_cleaned)
#             visualizations = generate_visualizations(df_cleaned)

#             # Return all results
#             return {
#                 "status": "success",
#                 "message": f"Image processed successfully. Output saved as {output_format}",
#                 "filename": os.path.basename(output_path),
#                 "path": f"/uploaded/{os.path.basename(output_path)}",
#                 "summary": summary,
#                 "preview": preview,
#                 "statistics": stats,
#                 "insights": insights,
#                 "visualizations": visualizations
#             }
#         else:
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "status": "error",
#                     "message": "Failed to process image"
#                 }
#             )
            
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status": "error",
#                 "message": f"Error processing image: {str(e)}"
#             }
#         )
