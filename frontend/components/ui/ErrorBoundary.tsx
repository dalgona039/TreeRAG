"use client";

import { Component, type ReactNode, type ErrorInfo } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-6 bg-gray-800 rounded-lg border border-gray-700">
          <AlertTriangle className="w-12 h-12 text-yellow-500 mb-4" />
          <h2 className="text-lg font-semibold text-gray-200 mb-2">
            Something went wrong
          </h2>
          <p className="text-sm text-gray-400 mb-4 text-center max-w-md">
            {this.state.error?.message ?? "An unexpected error occurred"}
          </p>
          <button
            onClick={this.handleRetry}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

interface QueryErrorBoundaryProps {
  children: ReactNode;
  resetKeys?: unknown[];
}

export function QueryErrorBoundary({ children, resetKeys }: QueryErrorBoundaryProps) {
  return (
    <ErrorBoundary
      key={resetKeys?.join("-")}
      fallback={
        <div className="flex flex-col items-center justify-center p-6 text-gray-400">
          <AlertTriangle className="w-8 h-8 mb-2 text-yellow-500" />
          <p>Failed to load data</p>
        </div>
      }
    >
      {children}
    </ErrorBoundary>
  );
}
