import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';
import { AppService } from '../../services/app.service';
import { map, take } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class DashboardGuard implements CanActivate {
  constructor(private appService: AppService, private router: Router) {}

  // ... existing code ...
  canActivate() {
    return this.appService.getUploadData().pipe(
      take(1),
      map(data => {
        if (data) {
          return true;
        } else {
          // Redirect to upload page if no data is available
          this.router.navigate(['/upload-file']);
          return false;
        }
      })
    );
  }
} 