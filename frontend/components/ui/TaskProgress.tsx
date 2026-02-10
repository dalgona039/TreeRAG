"use client";

import { useEffect, useState } from "react";
import { useTaskStatus, useCancelTask } from "@/hooks/useQueries";
import { Loader2, CheckCircle2, XCircle, AlertCircle, X } from "lucide-react";
import { toast } from "react-hot-toast";

interface TaskProgressProps {
  taskId: string;
  onComplete?: (result: unknown) => void;
  onError?: (error: string) => void;
  onCancel?: () => void;
  showCancelButton?: boolean;
}

export function TaskProgress({
  taskId,
  onComplete,
  onError,
  onCancel,
  showCancelButton = true,
}: TaskProgressProps) {
  const [isComplete, setIsComplete] = useState(false);
  
  const { data: status, isLoading, error } = useTaskStatus(
    isComplete ? null : taskId,
    { refetchInterval: 2000 }
  );
  
  const cancelMutation = useCancelTask();
  
  useEffect(() => {
    if (!status) return;
    
    if (status.state === "SUCCESS") {
      setIsComplete(true);
      onComplete?.(status.result);
    } else if (status.state === "FAILURE") {
      setIsComplete(true);
      onError?.(status.error ?? "Task failed");
    } else if (status.state === "REVOKED") {
      setIsComplete(true);
      onCancel?.();
    }
  }, [status, onComplete, onError, onCancel]);
  
  const handleCancel = async () => {
    try {
      await cancelMutation.mutateAsync(taskId);
      toast.success("Task cancelled");
      onCancel?.();
    } catch {
      toast.error("Failed to cancel task");
    }
  };
  
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-gray-400">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Loading task status...</span>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex items-center gap-2 text-red-400">
        <AlertCircle className="w-4 h-4" />
        <span>Failed to load task status</span>
      </div>
    );
  }
  
  if (!status) return null;
  
  const getStateIcon = () => {
    switch (status.state) {
      case "SUCCESS":
        return <CheckCircle2 className="w-5 h-5 text-green-400" />;
      case "FAILURE":
        return <XCircle className="w-5 h-5 text-red-400" />;
      case "REVOKED":
        return <X className="w-5 h-5 text-gray-400" />;
      default:
        return <Loader2 className="w-5 h-5 animate-spin text-blue-400" />;
    }
  };
  
  const getProgressBar = () => {
    if (!status.progress) return null;
    
    const progress = status.progress.progress ?? 0;
    
    return (
      <div className="w-full mt-2">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>{status.progress.stage}</span>
          <span>{progress}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        {status.progress.message && (
          <p className="text-xs text-gray-500 mt-1">{status.progress.message}</p>
        )}
      </div>
    );
  };
  
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {getStateIcon()}
          <div>
            <p className="font-medium text-gray-200">
              {status.state === "PENDING" && "Waiting in queue..."}
              {status.state === "PROGRESS" && "Processing..."}
              {status.state === "SUCCESS" && "Completed"}
              {status.state === "FAILURE" && "Failed"}
              {status.state === "REVOKED" && "Cancelled"}
            </p>
            <p className="text-xs text-gray-500">Task ID: {taskId.slice(0, 8)}...</p>
          </div>
        </div>
        
        {showCancelButton && !isComplete && (
          <button
            onClick={handleCancel}
            disabled={cancelMutation.isPending}
            className="px-3 py-1 text-sm text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded transition-colors"
          >
            {cancelMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              "Cancel"
            )}
          </button>
        )}
      </div>
      
      {getProgressBar()}
      
      {status.state === "FAILURE" && status.error && (
        <div className="mt-2 p-2 bg-red-900/20 border border-red-800 rounded text-sm text-red-400">
          {status.error}
        </div>
      )}
      
      {status.state === "SUCCESS" && status.result && (
        <div className="mt-2 p-2 bg-green-900/20 border border-green-800 rounded text-sm text-green-400">
          {status.result.index_filename
            ? `Indexed: ${status.result.index_filename}`
            : "Task completed successfully"}
        </div>
      )}
    </div>
  );
}
