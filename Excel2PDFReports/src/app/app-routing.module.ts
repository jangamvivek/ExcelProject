import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './shared/components/dashboard/dashboard.component';
import { FilePreviewComponent } from './shared/components/file-preview/file-preview.component';
import { UploadFileComponent } from './shared/components/upload-file/upload-file.component';

const routes: Routes = [
  { path: 'upload-file', component: UploadFileComponent},
  { path: 'file-preview', component: FilePreviewComponent},
  { path: 'dashboard', component: DashboardComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
