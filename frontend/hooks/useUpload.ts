import { useState, useRef } from "react";
import { toast } from "react-hot-toast";
import { api } from "@/lib/api";
import type { ChatSession, UploadProgressState, ApiError } from "@/lib/types";

export function useUpload(
  setSessions: React.Dispatch<React.SetStateAction<ChatSession[]>>,
  setCurrentSessionId: (id: string) => void
) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgressState | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileUploadAndIndex = async (
    e: React.ChangeEvent<HTMLInputElement>,
    t: any
  ) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const files = Array.from(e.target.files);
    const totalFiles = files.length;
    
    try {
      setIsUploading(true);

      const indexFiles: string[] = [];
      const docNames: string[] = [];
      const originalFilenames: string[] = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        setUploadProgress({
          current: i + 1,
          total: totalFiles,
          currentFile: file.name,
          status: 'uploading'
        });
        
        const uploadRes = await api.uploadFile(file);
        const uploadedFilename = uploadRes.data.filename;

        setUploadProgress({
          current: i + 1,
          total: totalFiles,
          currentFile: file.name,
          status: 'indexing'
        });
        
        const indexRes = await api.indexFile(uploadedFilename);
        
        indexFiles.push(indexRes.data.index_file);
        docNames.push(file.name.replace('.pdf', ''));
        originalFilenames.push(file.name);
      }

      const sessionTitle = files.length === 1 
        ? docNames[0] 
        : `${docNames[0]} 외 ${files.length - 1}건`;

      const newSession: ChatSession = {
        id: Date.now().toString(),
        title: sessionTitle,
        indexFiles: indexFiles,
        originalFilenames: originalFilenames, // 원본 파일명 추가
        messages: [{ 
          role: "assistant", 
          content: `반갑습니다! ${files.length}개 문서(${docNames.join(", ")})에 대한 분석 준비가 완료되었습니다. 무엇이든 물어보세요.` 
        }],
        createdAt: new Date(),
      };

      setSessions(prev => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      
      setUploadProgress({
        current: totalFiles,
        total: totalFiles,
        currentFile: '',
        status: 'complete'
      });
      
      setTimeout(() => setUploadProgress(null), 2000);
      toast.success(t.analysisComplete);
    } catch (error) {
      const err = error as { response?: { data?: ApiError } };
      const message = err.response?.data?.detail || t.uploadFailed;
      toast.error(message);
      console.error(error);
      setUploadProgress(null);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return {
    isUploading,
    uploadProgress,
    fileInputRef,
    handleFileUploadAndIndex,
  };
}
