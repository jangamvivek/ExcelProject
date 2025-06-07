from fastapi import FastAPI, UploadFile, File, Query
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
import shutil
from scraper import process_image

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    filename = f"correlation_heatmap_{uuid.uuid4().hex[:8]}.png"
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
    
    filename = f"histogram_{col}_{uuid.uuid4().hex[:8]}.png"
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
    
    filename = f"bar_chart_{col}_{uuid.uuid4().hex[:8]}.png"
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
    
    filename = f"pie_chart_{col}_{uuid.uuid4().hex[:8]}.png"
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
        
        filename = f"boxplot_{col}_{uuid.uuid4().hex[:8]}.png"
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

        summary = {
            "rows": df_cleaned.shape[0],
            "columns": df_cleaned.shape[1],
            "columns_list": df_cleaned.columns.tolist(),
            "missing_values": int(df_cleaned.isnull().sum().sum())
        }

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


        insights = generate_insights(df_cleaned)
        visualizations = generate_visualizations(df_cleaned)

        return {
            "filename": file.filename,
            "summary": summary,
            "preview": preview,
            "statistics": stats,
            "insights": insights,
            "visualizations": visualizations
        }

    except Exception as e:
        return {"error": str(e)}


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
            # Process the image using OCR (default to Excel format)
            output_path = process_image(file_path, 'excel')
            
            if not output_path:
                raise Exception("Failed to generate output file")
            
            # Read the generated Excel file
            df = pd.read_excel(output_path)
            
            # Clean the data
            df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
            preview = df_cleaned.head().to_dict(orient="records")

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

            # Generate insights and visualizations
            insights = generate_insights(df_cleaned)
            visualizations = generate_visualizations(df_cleaned)

            # Return all results
            return {
                "status": "success",
                "message": "✅ Image processed successfully",
                "filename": os.path.basename(output_path),
                "path": f"/uploaded/{os.path.basename(output_path)}",
                "summary": summary,
                "preview": preview,
                "statistics": stats,
                "insights": insights,
                "visualizations": visualizations
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

@app.post("/process-image")
async def process_image_endpoint(
    file: UploadFile = File(...),
    output_format: str = Query("excel", enum=["excel", "csv"])
):
    """
    Process an uploaded image to extract text and convert to Excel/CSV format
    """
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the image using OCR
        output_path = process_image(file_path, output_format)
        
        if output_path:
            # Read the generated Excel/CSV file
            if output_format == 'excel':
                df = pd.read_excel(output_path)
            else:
                df = pd.read_csv(output_path)
            
            # Clean the data (same as your existing workflow)
            df_cleaned = df.dropna(how="all").dropna(axis=1, how="all")
            preview = df_cleaned.head().to_dict(orient="records")

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

            # Generate insights and visualizations
            insights = generate_insights(df_cleaned)
            visualizations = generate_visualizations(df_cleaned)

            # Return all results
            return {
                "status": "success",
                "message": f"Image processed successfully. Output saved as {output_format}",
                "filename": os.path.basename(output_path),
                "path": f"/uploaded/{os.path.basename(output_path)}",
                "summary": summary,
                "preview": preview,
                "statistics": stats,
                "insights": insights,
                "visualizations": visualizations
            }
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Failed to process image"
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error processing image: {str(e)}"
            }
        )
