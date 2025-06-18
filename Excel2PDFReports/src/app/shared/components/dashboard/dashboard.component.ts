import { Component, OnInit, OnDestroy, ViewChild, TemplateRef } from '@angular/core';
import { AppService } from '../../../services/app.service';
import { UploadResponse } from '../../../models/data.interface';
import { Subscription } from 'rxjs';
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { environment } from '../../../../environments/environment';

type StatisticValue = {
  sum: number;
  mean: number;
  min: number;
  max: number;
  median: number;
  std_dev: number;
};

@Component({
  selector: 'app-dashboard',
  standalone: false,
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit, OnDestroy {
  // uploadData: UploadResponse | null = null;
  uploadData: any; // Temporary fix for dynamic structures

  private subscription: Subscription = new Subscription();
  private readonly apiUrl = environment.apiUrl;
  isSidebarOpen: boolean = false;
  headerTitle: any;
  headerDescription = '';
  selectedSize = 'title-medium';
  selectedAlignment = 'center';
  defaultBackground = 'linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%)';
  selectedColor: string = this.defaultBackground;


  protected readonly StatisticValueType = {
    sum: 0,
    mean: 0,
    min: 0,
    max: 0,
    median: 0,
    std_dev: 0
  } as StatisticValue;

  bgColors: string[] = [
    'transparent', // No color
    '#ffffff',     // White
    '#e0e0e0',     // Light gray
  
    '#000000',     // Black
    'linear-gradient(90deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%)', // Dark gradient
    'linear-gradient(90deg, #004e92 0%, #000428 100%)',              // Blue gradient
    
    'linear-gradient(90deg, #ff512f 0%, #dd2476 100%)',              // Red gradient
    'linear-gradient(90deg, #e1eec3 0%, #f05053 100%)',              // Soft pink/yellow
    
    '#2a86f7',                                                       // Primary
    'linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%)',              // Pink/blue
  ];
  
  getTitleColor(): string {
    const darkBackgrounds = [
      '#000000',
      '#2a86f7',           
      'linear-gradient(90deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%)',
      'linear-gradient(90deg, #e1eec3 0%, #f05053 100%)', 
      'linear-gradient(90deg, #004e92 0%, #000428 100%)',  
      'linear-gradient(90deg, #ff512f 0%, #dd2476 100%)',
    ];
  
    return darkBackgrounds.includes(this.selectedColor) ? 'white' : 'black';
  }
  
  private chartColors = [
    '#ffacac', //'#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    // '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    // '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
  ];
  
  @ViewChild('editHeaderModal') editHeaderModal: any;

  constructor(private appService: AppService, private modalService: NgbModal) {}

  ngOnInit(): void {
    debugger
    this.subscription = this.appService.getUploadData().subscribe((data:any) => {
      debugger
      if (data) {
        this.uploadData = data;
        this.headerTitle = data?.dashboard.header.title;
        console.log('Dashboard received data:', data);
      } else {
        console.log('No upload data yet.');
      }
    });
  }
  

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  toggleSidebar(): void {
    this.isSidebarOpen = !this.isSidebarOpen;
  }

  openEditModal() {
    this.modalService.open(this.editHeaderModal, { size: 'lg', centered: true });
  }

  openEditHeader(modal: TemplateRef<any>) {
    this.modalService.open(modal, {
      windowClass: 'no-animate',
      backdrop: true,
      centered: false,
      scrollable: true
    });
  }
  
  selectSize(size: string) {
    this.selectedSize = `title-${size}`;
  }

  getFontSizeClass(): string {
    return `title-${this.selectedSize}`;
  }

  selectAlignment(alignment: string): void {
    this.selectedAlignment = alignment;
  }

  
  saveChanges(modal: any) {
    // save logic (e.g. update backend or UI)
    modal.close();
  }

  // Helper methods to check if data exists
  hasData(): boolean {
    return this.uploadData !== null;
  }

  getSummary(): any {
    return this.uploadData?.summary;
  }

  getStatistics(): { [key: string]: StatisticValue } | undefined {
    return this.uploadData?.statistics as { [key: string]: StatisticValue } | undefined;
  }

  // Helper method to safely get statistic value
  getStatValue(stat: any, field: keyof StatisticValue): number {
    return stat?.value?.[field] ?? 0;
  }

  getInsights(): string[] {
    return this.uploadData?.insights || [];
  }

  getChartData(): any {
    return this.uploadData?.chart_data;
  }

  getPreviewData(): any[] {
    return this.uploadData?.preview || [];
  }

  // Get full URLs for visualizations
  // getVisualizationUrls(): string[] {
  //   const viz = this.uploadData?.visualizations;
  
  //   if (Array.isArray(viz)) {
  //     return viz.map(item => `${this.apiUrl}${item}`);
  //   }
  
  //   if (typeof viz === 'string') {
  //     return [`${this.apiUrl}${viz}`];
  //   }
  
  //   if (viz && typeof viz === 'object') {
  //     return Object.values(viz).map(item => `${this.apiUrl}${item}`);
  //   }
  
  //   return [];
  // }
  getBarChartSeries(chart: any) {
    return [{
      name: chart.column,
      data: chart.data.values.map((value: any, index: number) => ({
        x: chart.data.labels[index],
        y: value,
        fillColor: this.getColorForIndex(index)
      }))
    }];
  }

  
  getSeries(chart: any) {
    return [{
      name: chart.column,
      data: chart.data.values
    }];
  }
  
  // Box Plot Methods
getBoxPlotSeries(chart: any) {
  console.log('BoxPlot chart.data:', chart.data);
  return Array.isArray(chart.data)
    ? chart.data.map((d: any) => ({
        x: d.label,
        y: [d.min, d.q1, d.median, d.q3, d.max]
      }))
    : [];
}

getBoxPlotMedian(chart: any): string {
  if (Array.isArray(chart.data) && chart.data.length > 0) {
    return chart.data[0].median?.toFixed(2) || 'N/A';
  }
  return 'N/A';
}

getBoxPlotRange(chart: any): string {
  if (Array.isArray(chart.data) && chart.data.length > 0) {
    const data = chart.data[0];
    return `${data.min?.toFixed(2)} - ${data.max?.toFixed(2)}` || 'N/A';
  }
  return 'N/A';
}
  
  // Generate colors for bar chart
getBarColors(labels: string[]): string[] {
  return labels.map((_, index) => this.getColorForIndex(index));
}

// Generate colors for X-axis labels
getXAxisLabelColors(labels: string[]): string[] {
  return labels.map((_, index) => this.getColorForIndex(index));
}
// Pie Chart Methods
getPieColors(labels: string[]): string[] {
  return labels.map((_, index) => this.getColorForIndex(index));
}



// Get color for specific index
getColorForIndex(index: number): string {
  return this.chartColors[index % this.chartColors.length];
}
  
  // Heatmap Methods
getHeatmapSeries(correlation: any) {
  const series = [];
  const columns = correlation.columns;
  const data = correlation.data;

  for (let i = 0; i < columns.length; i++) {
    series.push({
      name: columns[i],
      data: data[i].map((val: number, idx: number) => ({
        x: columns[idx],
        y: val
      }))
    });
  }
  return series;
}
  
  // Histogram Methods
  getHistogramSeries(chart: any) {
    return [{
      name: chart.column,
      data: chart.data.values.map((value: any, index: number) => ({
        x: this.getHistogramLabels(chart)[index],
        y: value,
        fillColor: this.getColorForIndex(index)
      }))
    }];
  }

  getHistogramColors(chart: any): string[] {
    return chart.data.values.map((_: any, index: number) => this.getColorForIndex(index));
  }

  getHistogramLabelColors(chart: any): string[] {
    return chart.data.values.map((_: any, index: number) => this.getColorForIndex(index));
  }
  
  getHistogramLabels(chart: any) {
    const bins = chart.data.bins;
    const labels = [];
    for (let i = 0; i < bins.length - 1; i++) {
      labels.push(`${Math.round(bins[i])}-${Math.round(bins[i + 1])}`);
    }
    return labels;
  }
  
}
