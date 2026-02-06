import { Loader2 } from "lucide-react";
import type { UploadProgressState } from "@/lib/types";

interface UploadProgressProps {
  uploadProgress: UploadProgressState;
  t: {
    uploading: string;
    indexing: string;
    complete: string;
    files: string;
  };
}

export default function UploadProgress({ uploadProgress, t }: UploadProgressProps) {
  return (
    <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-b border-green-200 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Loader2 className="animate-spin text-emerald-600" size={16} />
            <span className="text-sm font-semibold text-slate-800">
              {uploadProgress.status === 'uploading' && t.uploading}
              {uploadProgress.status === 'indexing' && t.indexing}
              {uploadProgress.status === 'complete' && t.complete}
            </span>
          </div>
          <span className="text-xs text-slate-600">
            {uploadProgress.current} / {uploadProgress.total} {t.files}
          </span>
        </div>
        <div className="bg-white rounded-full h-2 overflow-hidden mb-2">
          <div 
            className="bg-gradient-to-r from-emerald-500 to-green-500 h-full transition-all duration-300"
            style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
          />
        </div>
        {uploadProgress.currentFile && (
          <p className="text-xs text-slate-600 truncate">
            📄 {uploadProgress.currentFile}
          </p>
        )}
      </div>
    </div>
  );
}
