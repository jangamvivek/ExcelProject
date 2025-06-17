import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AppService } from '../../../services/app.service';
import { UploadResponse } from '../../../models/data.interface';
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
    const formData = new FormData();
    formData.append('file', file);

    this.http.post<UploadResponse>(`${environment.apiUrl}upload`, formData).subscribe({
      next: (response) => {
        console.log('✅ File uploaded successfully:', response);
        this.appService.setUploadData(response);
        this.isLoading = false;
        this.router.navigate(['/dashboard']);
      },
      error: (error) => {
        console.error('❌ Upload failed:', error);
        alert('Failed to upload file.');
        this.isLoading = false;
      }
    });
  }

  uploadSelectedFile(): void {
    if (this.selectedFile) {
      this.uploadFileToBackend(this.selectedFile);
    } else {
      alert('No file selected to upload.');
    }
  }
  
  // Send URL to backend for scraping
  uploadUrlToBackend(): void {
    if (!this.urlInput || !this.isValidUrl(this.urlInput)) {
      alert('Please enter a valid URL.');
      return;
    }
    this.isLoading = true;
    // Use GET with query parameter
    this.http.get(`${environment.apiUrl}scrape`, {
      params: { url: this.urlInput }
    }).subscribe({
      next: (response) => {
        console.log('✅ URL sent successfully:', response);
        alert('URL sent successfully!');
        this.urlInput = '';
        this.isLoading = false;
      },
      error: (error) => {
        console.error('❌ Sending URL failed:', error);
        alert('Failed to send URL.');
        this.isLoading = false;
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

  uploadImageToBackend(image: File): void {
    this.isLoading = true;
    const formData = new FormData();
    formData.append('file', image); // FastAPI should accept 'file'

    this.http.post(`${environment.apiUrl}upload-image`, formData).subscribe({
      next: (res) => {
        console.log('✅ Image uploaded:', res);
        alert('Image uploaded successfully!');
        this.isLoading = false;
      },
      error: (err) => {
        console.error('❌ Image upload failed:', err);
        alert('Image upload failed!');
        this.isLoading = false;
      }
    });
  }

  
  
}
