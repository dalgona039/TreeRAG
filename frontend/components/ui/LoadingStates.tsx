"use client";

import { Loader2 } from "lucide-react";
import { clsx } from "clsx";

interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

export function Spinner({ size = "md", className }: SpinnerProps) {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6",
    lg: "w-10 h-10",
  };

  return (
    <Loader2
      className={clsx(
        "animate-spin text-blue-500",
        sizeClasses[size],
        className
      )}
    />
  );
}

interface SkeletonProps {
  className?: string;
  variant?: "text" | "circular" | "rectangular";
  width?: string | number;
  height?: string | number;
}

export function Skeleton({ 
  className, 
  variant = "text",
  width,
  height,
}: SkeletonProps) {
  const baseClasses = "animate-pulse bg-gray-700";
  
  const variantClasses = {
    text: "rounded h-4",
    circular: "rounded-full",
    rectangular: "rounded-lg",
  };

  return (
    <div
      className={clsx(baseClasses, variantClasses[variant], className)}
      style={{
        width: width ?? (variant === "text" ? "100%" : undefined),
        height: height ?? (variant === "circular" ? width : undefined),
      }}
    />
  );
}

interface LoadingOverlayProps {
  isLoading: boolean;
  children: React.ReactNode;
  message?: string;
}

export function LoadingOverlay({ isLoading, children, message }: LoadingOverlayProps) {
  return (
    <div className="relative">
      {children}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900/80 flex flex-col items-center justify-center z-50 rounded-lg">
          <Spinner size="lg" />
          {message && (
            <p className="mt-3 text-sm text-gray-300">{message}</p>
          )}
        </div>
      )}
    </div>
  );
}

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      {icon && (
        <div className="mb-4 text-gray-500">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-medium text-gray-300 mb-1">{title}</h3>
      {description && (
        <p className="text-sm text-gray-500 mb-4 max-w-sm">{description}</p>
      )}
      {action}
    </div>
  );
}

interface InlineLoadingProps {
  text?: string;
}

export function InlineLoading({ text = "Loading..." }: InlineLoadingProps) {
  return (
    <div className="flex items-center gap-2 text-gray-400">
      <Spinner size="sm" />
      <span className="text-sm">{text}</span>
    </div>
  );
}

interface ListSkeletonProps {
  count?: number;
  itemHeight?: number;
}

export function ListSkeleton({ count = 5, itemHeight = 48 }: ListSkeletonProps) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton
          key={i}
          variant="rectangular"
          height={itemHeight}
          className="w-full"
        />
      ))}
    </div>
  );
}

interface CardSkeletonProps {
  showImage?: boolean;
}

export function CardSkeleton({ showImage = false }: CardSkeletonProps) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      {showImage && (
        <Skeleton 
          variant="rectangular" 
          height={160} 
          className="w-full mb-4" 
        />
      )}
      <Skeleton variant="text" className="w-3/4 mb-2" />
      <Skeleton variant="text" className="w-1/2 mb-4" />
      <div className="flex gap-2">
        <Skeleton variant="rectangular" width={60} height={24} />
        <Skeleton variant="rectangular" width={60} height={24} />
      </div>
    </div>
  );
}
