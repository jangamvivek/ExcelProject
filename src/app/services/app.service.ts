import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { 
  EnhancedVisualizationData, 
  FileUploadResponse, 
  VisualizationRecommendation,
  DistributionData,
  CategoricalData,
  TimeSeriesData,
  BoxPlotData,
  OutlierAnalysis
} from '../models/data.interface';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AppService {
  private apiUrl = environment.apiUrl;
  private uploadDataSubject: BehaviorSubject<FileUploadResponse | null>;
  public uploadData$: Observable<FileUploadResponse | null>;
  private chartDataSubject = new BehaviorSubject<any>(null);
  public chartData$ = this.chartDataSubject.asObservable();

  constructor(private http: HttpClient) {
    let initialData: FileUploadResponse | null = null;
  
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const storedData = localStorage.getItem('uploadData');
      if (storedData) {
        try {
          initialData = JSON.parse(storedData);
        } catch (e) {
          console.error('Error parsing stored upload data:', e);
          localStorage.removeItem('uploadData'); // Clear invalid data
        }
      }
    }
  
    this.uploadDataSubject = new BehaviorSubject<FileUploadResponse | null>(initialData);
    this.uploadData$ = this.uploadDataSubject.asObservable();
  }
  

  uploadFile(file: File): Observable<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<FileUploadResponse>(`${this.apiUrl}/upload`, formData);
  }

  updateChartData(data: any) {
    this.chartDataSubject.next(data);
  }

  setUploadData(data: FileUploadResponse): void {
    this.uploadDataSubject.next(data);
    localStorage.setItem('uploadData', JSON.stringify(data));
  }

  getUploadData(): Observable<FileUploadResponse | null> {
    return this.uploadData$;
  }
}
