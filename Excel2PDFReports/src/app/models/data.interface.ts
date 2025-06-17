 // src/app/models/data.interface.ts
export interface ChartDataPoint {
  x: string | number;
  y: number;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[] | number[];
}

export interface ChartData {
  bar_charts: ChartSeries[];
  pie_charts: ChartSeries[];
  line_charts: ChartSeries[];
  heatmap_data: {
      x: string[];
      y: string[];
      data: number[][];
  } | null;
}

export interface UploadResponse {
  status: string;
  message: string;
  filename: string;
  path: string;
  summary: {
      rows: number;
      columns: number;
      columns_list: string[];
      missing_values: number;
  };
  preview: any[];
  statistics: {
      [key: string]: {
          sum: number;
          mean: number;
          min: number;
          max: number;
          median: number;
          std_dev: number;
      };
  };
  insights: string[];
  visualizations: string[];
  chart_data: ChartData;
  raw_data: {
      columns: string[];
      data: any[];
  };
}

interface Visualization {
  bar_charts?: any[];
  pie_charts?: any[];
  box_plots?: any[];
  histograms?: any[];
  time_series?: any[];
  correlation?: {
    columns: string[];
    data: number[][];
    type: string;
  };
}

interface DashboardData {
  dashboard: {
    header: {
      title: string;
      subtitle?: string;
    };
  };
  visualizations: Visualization;
}
