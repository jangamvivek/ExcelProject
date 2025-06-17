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
    'linear-gradient(90deg, #000428 0%, #004e92 100%)',              // Blue gradient
    
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
      'linear-gradient(90deg, #000428 0%, #004e92 100%)',
      'linear-gradient(90deg, #ff512f 0%, #dd2476 100%)',
    ];
  
    return darkBackgrounds.includes(this.selectedColor) ? 'white' : 'black';
  }
  
  
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
      data: chart.data.values // or chart.data.percentages based on what you want to show
    }];
  }

  
  getSeries(chart: any) {
    return [{
      name: chart.column,
      data: chart.data.values
    }];
  }
  
  getBoxPlotSeries(chart: any) {
    console.log('BoxPlot chart.data:', chart.data);
    return Array.isArray(chart.data)
      ? chart.data.map((d: any) => ({
          x: d.label,
          y: [d.min, d.q1, d.median, d.q3, d.max]
        }))
      : [];
  }
  
  
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
  
  getHistogramSeries(chart: any) {
    return [{
      name: chart.column,
      data: chart.data.values
    }];
  }
  
  getHistogramLabels(chart: any) {
    // Convert bin edges into readable label ranges like "5000-6000"
    const bins = chart.data.bins;
    const labels = [];
  
    for (let i = 0; i < bins.length - 1; i++) {
      labels.push(`${Math.round(bins[i])}-${Math.round(bins[i + 1])}`);
    }
  
    return labels;
  }
  
  
}
