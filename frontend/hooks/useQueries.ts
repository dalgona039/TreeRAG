import { useQuery, useMutation, useQueryClient, UseQueryOptions } from "@tanstack/react-query";
import { api, API_BASE_URL } from "@/lib/api";
import axios from "axios";
import type { TreeData } from "@/lib/types";

export const queryKeys = {
  indices: ["indices"] as const,
  pdfs: ["pdfs"] as const,
  tree: (filename: string) => ["tree", filename] as const,
  cacheStats: ["cache", "stats"] as const,
  taskStatus: (taskId: string) => ["task", taskId] as const,
  activeTasks: ["tasks", "active"] as const,
};

export function useIndices() {
  return useQuery({
    queryKey: queryKeys.indices,
    queryFn: async () => {
      const res = await axios.get<{ indices: string[] }>(`${API_BASE_URL}/indices`);
      return res.data.indices;
    },
    staleTime: 30 * 1000,
  });
}

export function usePdfs() {
  return useQuery({
    queryKey: queryKeys.pdfs,
    queryFn: async () => {
      const res = await axios.get<{ pdfs: string[] }>(`${API_BASE_URL}/pdfs`);
      return res.data.pdfs;
    },
    staleTime: 30 * 1000,
  });
}

export function useTree(indexFilename: string | null) {
  return useQuery({
    queryKey: queryKeys.tree(indexFilename ?? ""),
    queryFn: () => api.loadTree(indexFilename!),
    enabled: !!indexFilename,
    staleTime: 5 * 60 * 1000,
  });
}

export function useCacheStats() {
  return useQuery({
    queryKey: queryKeys.cacheStats,
    queryFn: async () => {
      const res = await axios.get(`${API_BASE_URL}/cache/stats`);
      return res.data.cache_stats;
    },
    staleTime: 10 * 1000,
    refetchInterval: 30 * 1000,
  });
}

interface UploadResult {
  message: string;
  filename: string;
  original_filename: string;
  path: string;
  size_bytes: number;
}

export function useUploadFile() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (file: File) => {
      const res = await api.uploadFile(file);
      return res.data as UploadResult;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.pdfs });
    },
  });
}

interface IndexResult {
  message: string;
  index_file: string;
  status: string;
}

export function useIndexFile() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (filename: string) => {
      const res = await api.indexFile(filename);
      return res.data as IndexResult;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.indices });
    },
  });
}

interface ChatParams {
  question: string;
  index_filenames: string[];
  use_deep_traversal: boolean;
  max_depth: number;
  max_branches: number;
  domain_template: string;
  language: string;
  node_context?: {
    id: string;
    title: string;
    page_ref?: string;
    summary?: string;
  };
}

interface ChatResponse {
  answer: string;
  citations: string[];
  comparison?: {
    has_comparison: boolean;
    documents_compared: string[];
    commonalities?: string;
    differences?: string;
  };
  traversal_info?: {
    used_deep_traversal: boolean;
    nodes_visited: string[];
    nodes_selected: Array<{
      title: string;
      page_ref?: string;
      score?: number;
    }>;
    max_depth: number;
    max_branches: number;
  };
  resolved_references?: Array<{
    title: string;
    page_ref?: string;
    summary?: string;
  }>;
  hallucination_warning?: {
    message: string;
    overall_confidence: number;
    threshold: number;
  };
}

export function useChat() {
  return useMutation({
    mutationFn: async (params: ChatParams) => {
      const res = await api.chat(params);
      return res.data as ChatResponse;
    },
  });
}

export function useClearCache() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async () => {
      const res = await axios.post(`${API_BASE_URL}/cache/clear`);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cacheStats });
    },
  });
}

interface TaskStatus {
  task_id: string;
  state: "PENDING" | "PROGRESS" | "SUCCESS" | "FAILURE" | "REVOKED";
  ready: boolean;
  successful?: boolean;
  progress?: {
    stage: string;
    progress: number;
    message?: string;
    current?: number;
    total?: number;
  };
  result?: {
    status: string;
    filename?: string;
    index_filename?: string;
    error?: string;
  };
  error?: string;
  message?: string;
}

export function useTaskStatus(taskId: string | null, options?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: queryKeys.taskStatus(taskId ?? ""),
    queryFn: async () => {
      const res = await axios.get<TaskStatus>(`${API_BASE_URL}/tasks/${taskId}`);
      return res.data;
    },
    enabled: !!taskId,
    refetchInterval: options?.refetchInterval ?? 2000,
    refetchIntervalInBackground: true,
  });
}

interface AsyncIndexResponse {
  task_id: string;
  status: string;
  message: string;
}

export function useAsyncIndex() {
  return useMutation({
    mutationFn: async (filename: string) => {
      const res = await axios.post<AsyncIndexResponse>(`${API_BASE_URL}/tasks/index`, { filename });
      return res.data;
    },
  });
}

export function useAsyncBatchIndex() {
  return useMutation({
    mutationFn: async (filenames: string[]) => {
      const res = await axios.post<AsyncIndexResponse>(`${API_BASE_URL}/tasks/index/batch`, { filenames });
      return res.data;
    },
  });
}

export function useCancelTask() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (taskId: string) => {
      const res = await axios.delete(`${API_BASE_URL}/tasks/${taskId}`);
      return res.data;
    },
    onSuccess: (_, taskId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.taskStatus(taskId) });
    },
  });
}

export function useActiveTasks() {
  return useQuery({
    queryKey: queryKeys.activeTasks,
    queryFn: async () => {
      const res = await axios.get(`${API_BASE_URL}/tasks/`);
      return res.data;
    },
    staleTime: 5 * 1000,
    refetchInterval: 10 * 1000,
  });
}
