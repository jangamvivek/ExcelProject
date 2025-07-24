import { Component } from '@angular/core';

interface CalculationResult {
  totalInvestment: number;
  maturityAmount: number;
  totalGains: number;
  monthlyData?: any[];
}

@Component({
  selector: 'app-file-preview',
  standalone: false,
  templateUrl: './file-preview.component.html',
  styleUrl: './file-preview.component.css',
})
export class FilePreviewComponent {

}
