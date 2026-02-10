"use client";

import { type ReactNode } from "react";
import { Toaster } from "react-hot-toast";
import { QueryProvider } from "./QueryProvider";

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  return (
    <QueryProvider>
      {children}
      <Toaster
        position="bottom-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: "#1f2937",
            color: "#f9fafb",
            border: "1px solid #374151",
          },
          success: {
            iconTheme: {
              primary: "#10b981",
              secondary: "#f9fafb",
            },
          },
          error: {
            iconTheme: {
              primary: "#ef4444",
              secondary: "#f9fafb",
            },
          },
        }}
      />
    </QueryProvider>
  );
}
