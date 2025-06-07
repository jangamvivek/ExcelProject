import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

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

  constructor(private http: HttpClient) {}

  ngOnInit(){

  }

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
  
    // Normalize type to lowercase for safer comparison
    const fileType = file.type.toLowerCase();
  
    // Sometimes file.type might be empty for some browsers/files, so fallback to extension check
    const fileName = file.name.toLowerCase();
    const isValidType = allowedTypes.includes(fileType) || fileName.endsWith('.csv') || fileName.endsWith('.xls') || fileName.endsWith('.xlsx');
  
    if (!isValidType) {
      alert('Only Excel or CSV files are allowed.');
      return;
    }
  
    this.selectedFileName = file.name;
    this.selectedFile = file;
  
    // Automatically send the file to the backend
    this.uploadFileToBackend(file);
  }
  

  uploadFileToBackend(file: File): void {
    const formData = new FormData();
    formData.append('file', file);

    this.http.post('http://localhost:8000/upload', formData).subscribe({
      next: (response) => {
        console.log('✅ File uploaded successfully:', response);
        // You can show success message or update UI here
      },
      error: (error) => {
        console.error('❌ Upload failed:', error);
        alert('Failed to upload file.');
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
  
    // Use GET with query parameter
    this.http.get('http://localhost:8000/scrape', {
      params: { url: this.urlInput }
    }).subscribe({
      next: (response) => {
        console.log('✅ URL sent successfully:', response);
        alert('URL sent successfully!');
        this.urlInput = '';
      },
      error: (error) => {
        console.error('❌ Sending URL failed:', error);
        alert('Failed to send URL.');
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
  

  handleUpload(): void {
    if (this.selectedFile) {
      this.uploadSelectedFile();
    } else if (this.urlInput.trim()) {
      this.uploadUrlToBackend();
    } else {
      alert('Select a file or enter a URL.');
    }
  }

  handleImageUpload(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const image = input.files[0];
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    const imageType = image.type.toLowerCase();
    const imageName = image.name.toLowerCase();

    const isValidType = allowedTypes.includes(imageType) ||
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
    this.uploadImageToBackend(image);
  }

  uploadImageToBackend(image: File): void {
    const formData = new FormData();
    formData.append('file', image); // Ensure FastAPI endpoint expects 'image'

    this.http.post('http://localhost:8000/upload-image', formData).subscribe({
      next: (res) => {
        console.log('✅ Image uploaded:', res);
        alert('Image uploaded successfully!');
      },
      error: (err) => {
        console.error('❌ Image upload failed:', err);
        alert('Image upload failed!');
      }
    });
  }
  
}
