import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './shared/components/dashboard/dashboard.component';
import { FilePreviewComponent } from './shared/components/file-preview/file-preview.component';
import { UploadFileComponent } from './shared/components/upload-file/upload-file.component';
import { DashboardGuard } from './shared/guards/dashboard.guard';

const routes: Routes = [
  { path: '', redirectTo: 'upload-file', pathMatch: 'full'},
  { path: 'upload-file', component: UploadFileComponent},
  { path: 'file-preview', component: FilePreviewComponent},
  { 
    path: 'dashboard', 
    component: DashboardComponent,
    canActivate: [DashboardGuard]
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
