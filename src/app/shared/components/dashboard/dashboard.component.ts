import { Component, OnInit, OnDestroy, ViewChild, TemplateRef } from '@angular/core';
import { AppService } from '../../../services/app.service';
import { 
  EnhancedVisualizationData, 
  FileUploadResponse, 
  VisualizationRecommendation,
  DistributionData,
  CategoricalData,
  TimeSeriesData,
  BoxPlotData,
  OutlierAnalysis
} from '../../../models/data.interface';

import { Subscription } from 'rxjs';
import { NgbModal } from '@ng-bootstrap/ng-bootstrap';
import { environment } from '../../../../environments/environment';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';
import PptxGenJS from 'pptxgenjs';


type StatisticValue = {
  sum: number;
  mean: number;
  min: number;
  max: number;
  median: number;
  std_dev: number;
};

declare var bootstrap: any;
// declare var PptxGenJS: any;

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
  exportModal: any;

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

  openExportModal() {
    const modalElement = document.getElementById('exportModal');
    this.exportModal = new bootstrap.Modal(modalElement);
    this.exportModal.show();
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

  getInsightsLeft(): string[] {
    const insights = this.getInsights();
    const half = Math.ceil(insights.length / 2);
    return insights.slice(0, half);
  }
  
  getInsightsRight(): string[] {
    const insights = this.getInsights();
    const half = Math.ceil(insights.length / 2);
    return insights.slice(half);
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
// getBoxPlotSeries(chart: any) {
//   console.log('BoxPlot chart.data:', chart.data);
//   return Array.isArray(chart.data)
//     ? chart.data.map((d: any) => ({
//         x: d.label,
//         y: [d.min, d.q1, d.median, d.q3, d.max]
//       }))
//     : [];
// }
getBoxPlotSeries(chart: any) {
  if (!chart?.data) return [];
  
  return [{
    name: chart.column,
    data: [{
      x: chart.column,
      y: [
        chart.data.min,
        chart.data.q1,
        chart.data.median,
        chart.data.q3,
        chart.data.max
      ]
    }]
  }];
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
    async downloadChart(elementId: string, filename: string) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const canvas = await html2canvas(element, {
      scale: 2,
      logging: false,
      useCORS: true,
      allowTaint: true
    });
    
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }


async downloadDashboardAsPDF() {
  let originalTop:any
   let originalPosition:any
    let originalOverflow:any
    const doc = new jsPDF('p', 'mm', 'a4');
    const dashboardElement = document.getElementById('dashboard');
    
    if (!dashboardElement) {
      console.error('Dashboard element not found');
      return;
    }

    // Show loading indicator
    const loadingElement = document.createElement('div');
    loadingElement.style.position = 'fixed';
    loadingElement.style.top = '0';
    loadingElement.style.left = '0';
    loadingElement.style.width = '100%';
    loadingElement.style.height = '100%';
    loadingElement.style.backgroundColor = 'rgba(0,0,0,0.5)';
    loadingElement.style.display = 'flex';
    loadingElement.style.justifyContent = 'center';
    loadingElement.style.alignItems = 'center';
    loadingElement.style.zIndex = '10000';
    loadingElement.innerHTML = '<div style="color: white; font-size: 24px;">Generating PDF...</div>';
    document.body.appendChild(loadingElement);

    try {
      // Store original styles
       originalOverflow = document.body.style.overflow;
       originalPosition = dashboardElement.style.position;
       originalTop = dashboardElement.style.top;

      // Prepare for capture
      document.body.style.overflow = 'hidden';
      dashboardElement.style.position = 'absolute';
      dashboardElement.style.top = '0';

      const width = dashboardElement.scrollWidth;
      const fullHeight = dashboardElement.scrollHeight;
      
      // PDF page dimensions (with margins)
      const pdfPageWidth = doc.internal.pageSize.getWidth() - 20;
      const pdfPageHeight = doc.internal.pageSize.getHeight() - 20;
      
      // Calculate scale and how many pages we'll need
      const scale = pdfPageWidth / width;
      const scaledFullHeight = fullHeight * scale;
      const pagesNeeded = Math.ceil(scaledFullHeight / pdfPageHeight);
      const viewportHeight = pdfPageHeight / scale;
      
      // Create a container to hold our clone
      const captureContainer = document.createElement('div');
      captureContainer.style.position = 'absolute';
      captureContainer.style.left = '-9999px';
      captureContainer.style.width = `${width}px`;
      captureContainer.style.height = `${viewportHeight}px`;
      captureContainer.style.overflow = 'hidden';
      document.body.appendChild(captureContainer);
      
      // Clone the dashboard (only once)
      const dashboardClone:any = dashboardElement.cloneNode(true);
      dashboardClone.style.position = 'absolute';
      dashboardClone.style.top = '0';
      dashboardClone.style.left = '0';
      captureContainer.appendChild(dashboardClone);
      
      // Capture each page
      for (let page = 0; page < pagesNeeded; page++) {
        // Calculate scroll position for this page
        const scrollY = page * viewportHeight;
        dashboardClone.style.top = `-${scrollY}px`;
        
        // Capture this portion
        const canvas = await html2canvas(captureContainer, {
          scale: 2,
          logging: false,
          useCORS: true,
          windowWidth: width,
          windowHeight: viewportHeight
        });
        
        // Add new page if not first page
        if (page > 0) {
          doc.addPage();
        }
        
        // Add image to PDF
        doc.addImage(
          canvas.toDataURL('image/png'),
          'PNG',
          10, // x
          10, // y
          pdfPageWidth,
          Math.min(pdfPageHeight, scaledFullHeight - (page * pdfPageHeight)),
          undefined,
          'FAST'
        );
      }
      
      // Clean up clone container
      document.body.removeChild(captureContainer);
      
      // Generate and download PDF
      const pdfBlob = doc.output('blob');
      const url = window.URL.createObjectURL(pdfBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'dashboard-report.pdf';
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Error generating PDF. Please try again.');
    } finally {
      // Clean up
      if (document.body.contains(loadingElement)) {
        document.body.removeChild(loadingElement);
      }
      document.body.style.overflow = originalOverflow;
      dashboardElement.style.position = originalPosition;
      dashboardElement.style.top = originalTop;
    }
  }

  // Add this to your component class
  async downloadDashboardPDF() {
    await this.downloadDashboardAsPDF();
  }

 async downloadDashboardAsPPT() {debugger
    const dashboardElement = document.getElementById('dashboard');
    
    if (!dashboardElement) {
      console.error('Dashboard element not found');
      return;
    }

    // Show loading indicator
    const loadingElement = document.createElement('div');
    loadingElement.style.position = 'fixed';
    loadingElement.style.top = '0';
    loadingElement.style.left = '0';
    loadingElement.style.width = '100%';
    loadingElement.style.height = '100%';
    loadingElement.style.backgroundColor = 'rgba(0,0,0,0.5)';
    loadingElement.style.display = 'flex';
    loadingElement.style.justifyContent = 'center';
    loadingElement.style.alignItems = 'center';
    loadingElement.style.zIndex = '10000';
    loadingElement.innerHTML = '<div style="color: white; font-size: 24px;">Generating PowerPoint...</div>';
    document.body.appendChild(loadingElement);

    try {
      // Initialize PowerPoint
      const pptx = new PptxGenJS();
      pptx.layout = 'LAYOUT_16x9';
      
      // Find all chart containers (adjust selector as needed)
      const chartContainers:any = dashboardElement.querySelectorAll('.dashboard-title-card, .barChar-container,.pie-chart-container, .histogram-container,.boxplot-container,.heatmap-container,.insides');
      
      // Capture each chart individually
      for (const [index, chartContainer] of chartContainers.entries()) {
        // Create a temporary container to isolate the chart
        const tempContainer = document.createElement('div');
        tempContainer.style.position = 'absolute';
        tempContainer.style.left = '-9999px';
        tempContainer.style.width = `${chartContainer.offsetWidth}px`;
        tempContainer.style.height = `${chartContainer.offsetHeight}px`;
        tempContainer.style.overflow = 'hidden';
        document.body.appendChild(tempContainer);
        
        // Clone the chart
        const chartClone = chartContainer.cloneNode(true);
        tempContainer.appendChild(chartClone);
        
        // Capture the chart image
        const canvas = await html2canvas(chartClone, {
          scale: 2,
          logging: false,
          useCORS: true,
          windowWidth: chartContainer.offsetWidth,
          windowHeight: chartContainer.offsetHeight,
          backgroundColor: '#FFFFFF'
        });
        
        // Create new slide
        const slide = pptx.addSlide();
       // Calculate dimensions while maintaining aspect ratio
      const slideWidth = 10; // inches (16:9 slide width)
      const slideHeight = 5.625; // inches (16:9 slide height)
      const chartAspectRatio = chartContainer.offsetWidth / chartContainer.offsetHeight;
      const slideAspectRatio = slideWidth / slideHeight;
      
      let width, height, x, y;
      
      if (chartAspectRatio > slideAspectRatio) {
        // Chart is wider than slide - fit to width
        width = slideWidth;
        height = width / chartAspectRatio;
        x = 0;
        y = (slideHeight - height) / 2;
      } else {
       // Chart is taller than slide - fit to height
        height = slideHeight;
        width = height * chartAspectRatio;
        x = (slideWidth - width) / 2;
        y = 0;
      }
      
      // Add chart image (centered)
       
        slide.addImage({
          data: canvas.toDataURL('image/png'),
          x: x,
          y: y,
          w: width,
          h: height
        });
          
        // Remove temporary container
        document.body.removeChild(tempContainer);
      }
      
      // Download if we found charts
      if (chartContainers.length > 0) {
        pptx.writeFile({ fileName: 'dashboard-charts.pptx' });
      } else {
        alert('No charts found in the dashboard');
      }
      
    } catch (error) {
      console.error('Error generating PowerPoint:', error);
      alert('Error generating PowerPoint. Please try again.');
    } finally {
      // Clean up
      if (document.body.contains(loadingElement)) {
        document.body.removeChild(loadingElement);
      }
    }
  }
 
}
