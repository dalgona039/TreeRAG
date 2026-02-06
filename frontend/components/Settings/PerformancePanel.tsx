import { Activity } from "lucide-react";
import type { PerformanceMetrics } from "@/lib/types";

interface PerformancePanelProps {
  showPerformance: boolean;
  performanceMetrics: PerformanceMetrics;
  t: any;
}

export default function PerformancePanel({ 
  showPerformance, 
  performanceMetrics,
  t 
}: PerformancePanelProps) {
  if (!showPerformance) return null;

  return (
    <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border-b border-blue-200 p-4">
      <div className="max-w-4xl mx-auto">
        <h3 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
          <Activity size={16} className="text-blue-600" />
          {t.performance}
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <div className="text-xs text-slate-500 mb-1">{t.totalQueries}</div>
            <div className="text-2xl font-bold text-blue-600">{performanceMetrics.totalQueries}</div>
          </div>
          
          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <div className="text-xs text-slate-500 mb-1">{t.avgResponseTime}</div>
            <div className="text-2xl font-bold text-green-600">
              {performanceMetrics.avgResponseTime.toFixed(2)}{t.seconds}
            </div>
          </div>
          
          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <div className="text-xs text-slate-500 mb-1">{t.avgContextSize}</div>
            <div className="text-2xl font-bold text-purple-600">
              {Math.round(performanceMetrics.avgContextSize).toLocaleString()} {t.tokens}
            </div>
          </div>
          
          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <div className="text-xs text-slate-500 mb-1">{t.deepTraversalUsage}</div>
            <div className="text-2xl font-bold text-indigo-600">
              {performanceMetrics.deepTraversalUsage.toFixed(0)}%
            </div>
          </div>
        </div>
        
        {performanceMetrics.queriesHistory.length > 0 && (
          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <div className="text-xs font-medium text-slate-700 mb-2">{t.recentQueries}</div>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {performanceMetrics.queriesHistory.slice(-10).reverse().map((query, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-slate-600">
                    {new Date(query.timestamp).toLocaleTimeString()}
                  </span>
                  <div className="flex items-center gap-3">
                    <span className="text-green-600">{query.responseTime.toFixed(2)}{t.seconds}</span>
                    <span className="text-purple-600">{query.contextSize.toLocaleString()} {t.tokens}</span>
                    {query.useDeepTraversal && (
                      <span className="bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded text-xs">Deep</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
