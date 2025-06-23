import { HttpClient, HttpEvent, HttpEventType } from '@angular/common/http';
import { Component } from '@angular/core';
import { Router } from '@angular/router';
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
import { environment } from '../../../../environments/environment';

@Component({
  selector: 'app-upload-file',
  standalone: false,
  templateUrl: './upload-file.component.html',
  styleUrls: ['./upload-file.component.css']
})
export class UploadFileComponent {

  selectedFileName: string = '';
  selectedFile: File | null = null;
  urlInput: string = '';
  selectedImageName: string = '';
  selectedImage: File | null = null;

  constructor(
    private http: HttpClient, 
    private appService: AppService,
    private router: Router
  ) {}

  ngOnInit(){

  }

  isLoading = false;
  progressValue = 0;
  handleFileUpload(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) {
      console.warn('No file selected');
      return;
    }

    const file = input.files[0];
    const allowedTypes = [
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/csv'
    ];

    const fileType = file.type.toLowerCase();
    const fileName = file.name.toLowerCase();
    const isValidType =
      allowedTypes.includes(fileType) ||
      fileName.endsWith('.csv') ||
      fileName.endsWith('.xls') ||
      fileName.endsWith('.xlsx');

    if (!isValidType) {
      alert('Only Excel or CSV files are allowed.');
      return;
    }

    this.selectedFileName = file.name;
    this.selectedFile = file;
  }
  

  uploadFileToBackend(file: File): void {
    this.isLoading = true;
    this.progressValue = 0;
    const formData = new FormData();
    formData.append('file', file);
  
    this.http.post<FileUploadResponse>(`${environment.apiUrl}upload`, formData, {
      reportProgress: true,
      observe: 'events'
    }).subscribe({
      next: (event: HttpEvent<FileUploadResponse>) => {
        switch (event.type) {
          case HttpEventType.UploadProgress:
            if (event.total) {
              this.progressValue = Math.round((event.loaded / event.total) * 100);
              console.log(`Upload progress: ${this.progressValue}%`);
            }
            break;
          case HttpEventType.Response:
            console.log('✅ File uploaded successfully:', event.body);
            this.appService.setUploadData(event.body!);
            this.isLoading = false;
            this.progressValue = 100;
            this.router.navigate(['/dashboard']);
            break;
        }
      },
      error: (error) => {
        console.error('❌ Upload failed:', error);
        alert('Failed to upload file.');
        this.isLoading = false;
        this.progressValue = 0;
      }
    });
  }
  
  uploadImageToBackend(image: File): void {
    this.isLoading = true;
    this.progressValue = 0;
    const formData = new FormData();
    formData.append('file', image);
  
    this.http.post(`${environment.apiUrl}upload-image`, formData, {
      reportProgress: true,
      observe: 'events'
    }).subscribe({
      next: (event: HttpEvent<any>) => {
        switch (event.type) {
          case HttpEventType.UploadProgress:
            if (event.total) {
              this.progressValue = Math.round((event.loaded / event.total) * 100);
              console.log(`Image upload progress: ${this.progressValue}%`);
            }
            break;
          case HttpEventType.Response:
            console.log('✅ Image uploaded:', event.body);
            alert('Image uploaded successfully!');
            this.isLoading = false;
            this.progressValue = 100;
            break;
        }
      },
      error: (err) => {
        console.error('❌ Image upload failed:', err);
        alert('Image upload failed!');
        this.isLoading = false;
        this.progressValue = 0;
      }
    });
  }
  
  // For URL scraping (since it's not a file upload, simulate progress)
  uploadUrlToBackend(): void {
    if (!this.urlInput || !this.isValidUrl(this.urlInput)) {
      alert('Please enter a valid URL.');
      return;
    }
    
    this.isLoading = true;
    this.progressValue = 0;
    
    // Simulate progress for URL scraping
    const progressInterval = setInterval(() => {
      if (this.progressValue < 90) {
        this.progressValue += 10;
      }
    }, 300);
  
    this.http.get(`${environment.apiUrl}scrape`, {
      params: { url: this.urlInput }
    }).subscribe({
      next: (response) => {
        clearInterval(progressInterval);
        this.progressValue = 100;
        console.log('✅ URL sent successfully:', response);
        alert('URL sent successfully!');
        this.urlInput = '';
        setTimeout(() => {
          this.isLoading = false;
          this.progressValue = 0;
        }, 500);
      },
      error: (error) => {
        clearInterval(progressInterval);
        console.error('❌ Sending URL failed:', error);
        alert('Failed to send URL.');
        this.isLoading = false;
        this.progressValue = 0;
      }
    });
  }
  

  // Basic URL validation helper
  private isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }


  removeSelectedFile(): void {
    this.selectedFileName = '';
    this.selectedFile = null;
  }
  

  handleImageUpload(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const image = input.files[0];
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    const imageType = image.type.toLowerCase();
    const imageName = image.name.toLowerCase();

    const isValidType =
      allowedTypes.includes(imageType) ||
      imageName.endsWith('.png') ||
      imageName.endsWith('.jpg') ||
      imageName.endsWith('.jpeg') ||
      imageName.endsWith('.webp');

    if (!isValidType) {
      alert('Only image files (PNG, JPG, JPEG, WEBP) are allowed.');
      return;
    }

    this.selectedImageName = image.name;
    this.selectedImage = image;
  }


  handleUpload(): void {
    if (this.selectedFile) {
      this.uploadFileToBackend(this.selectedFile);
    }

    if (this.selectedImage) {
      this.uploadImageToBackend(this.selectedImage);
    }

    if (!this.selectedFile && !this.selectedImage && this.urlInput.trim()) {
      this.uploadUrlToBackend();
    }

    if (!this.selectedFile && !this.selectedImage && !this.urlInput.trim()) {
      alert('Select a file/image or enter a URL.');
    }
  }

  // uploadImageToBackend(image: File): void {
  //   this.isLoading = true;
  //   const formData = new FormData();
  //   formData.append('file', image); // FastAPI should accept 'file'

  //   this.http.post(`${environment.apiUrl}upload-image`, formData).subscribe({
  //     next: (res) => {
  //       console.log('✅ Image uploaded:', res);
  //       alert('Image uploaded successfully!');
  //       this.isLoading = false;
  //     },
  //     error: (err) => {
  //       console.error('❌ Image upload failed:', err);
  //       alert('Image upload failed!');
  //       this.isLoading = false;
  //     }
  //   });
  // }

  
}
