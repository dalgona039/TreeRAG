import { useState } from "react";
import type { PerformanceMetrics } from "@/lib/types";

export function usePerformance() {
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    totalQueries: 0,
    avgResponseTime: 0,
    avgContextSize: 0,
    deepTraversalUsage: 0,
    queriesHistory: []
  });

  return {
    performanceMetrics,
    setPerformanceMetrics,
  };
}
