// interfaces/data-analysis.interface.ts

export interface DataOverview {
  total_rows: number;
  total_columns: number;
  numeric_columns: number;
  categorical_columns: number;
  memory_usage_mb: number;
  completeness: number;
  data_quality_score: number;
}

export interface StatisticalMeasures {
  mean: number;
  median: number;
  mode: number;
  std: number;
  variance: number;
  skewness: number;
  kurtosis: number;
  min: number;
  max: number;
  range: number;
  q1: number;
  q3: number;
  iqr: number;
}

export interface DistributionShape {
  shape_description: string;
  tail_description: string;
  is_normal: boolean;
  skewness_interpretation: string;
  kurtosis_interpretation: string;
}

export interface DistributionData {
  column: string;
  type: string;
  data: {
    values: number[];
    bins: number[];
    bin_centers: number[];
    statistics: StatisticalMeasures;
    distribution_shape: DistributionShape;
    sample_size: number;
  };
}

export interface DiversityMetrics {
  unique_count: number;
  entropy: number;
  concentration_ratio: number;
  top_3_concentration: number;
}

export interface CategoricalData {
  column: string;
  type: string;
  data: {
    labels: string[];
    values: number[];
    percentages: number[];
    cumulative_percentages: number[];
    diversity_metrics: DiversityMetrics;
  };
  pie_data?: {
    labels: string[];
    values: number[];
    colors: string[];
  };
}

export interface CorrelationData {
  columns: string[];
  matrix: number[][];
  strong_correlations: Array<{
    col1: string;
    col2: string;
    correlation: number;
    strength: string;
  }>;
  type: string;
}

export interface TrendAnalysis {
  trend: string;
  strength: string;
  slope: number;
  r_squared: number;
}

export interface TimeSeriesData {
  date_column: string;
  value_column: string;
  type: string;
  data: {
    dates: string[];
    values: number[];
    frequency: string;
    trend_analysis: TrendAnalysis;
    data_points: number;
    date_range: {
      start: string;
      end: string;
    };
  };
}

export interface BoxPlotData {
  column: string;
  type: string;
  data: {
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    mean: number;
    outliers: number[];
    extreme_outliers: number[];
    outlier_count: number;
    extreme_outlier_count: number;
    whisker_low: number;
    whisker_high: number;
  };
}

export interface OutlierAnalysis {
  column: string;
  outlier_count: number;
  outlier_percentage: number;
  outlier_threshold: {
    lower: number;
    upper: number;
  };
  severity: string;
}

export interface VisualizationRecommendation {
  type: string;
  title: string;
  description: string;
  priority: string;
}

export interface StatisticalSummary {
  numeric_summary: { [key: string]: any };
  categorical_summary: { [key: string]: any };
}

export interface EnhancedVisualizationData {
  overview: DataOverview;
  correlation: CorrelationData | null;
  distributions: DistributionData[];
  categorical_analysis: CategoricalData[];
  comparative_analysis: BoxPlotData[];
  time_series: TimeSeriesData[];
  statistical_summary: StatisticalSummary;
  outlier_analysis: OutlierAnalysis[];
  recommendations: VisualizationRecommendation[];
}

export interface FileUploadResponse {
  status: string;
  message: string;
  timestamp: string;
  file_info: {
    filename: string;
    path: string;
    size_mb: number;
  };
  dashboard: any;
  summary: {
    original_rows: number;
    original_columns: number;
    cleaned_rows: number;
    cleaned_columns: number;
    columns_list: string[];
    missing_values: number;
    file_size_mb: number;
  };
  column_info: { [key: string]: any };
  preview: any[];
  statistics: { [key: string]: any };
  insights: any;
  visualizations: EnhancedVisualizationData;
  raw_data: {
    columns: string[];
    data: any[];
  };
}