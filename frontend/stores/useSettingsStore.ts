import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  // Traversal settings
  useDeepTraversal: boolean;
  maxDepth: number;
  maxBranches: number;
  
  // Display settings
  domainTemplate: string;
  language: "ko" | "en" | "ja";
  
  // Actions
  setUseDeepTraversal: (value: boolean) => void;
  setMaxDepth: (value: number) => void;
  setMaxBranches: (value: number) => void;
  setDomainTemplate: (value: string) => void;
  setLanguage: (value: "ko" | "en" | "ja") => void;
  
  // Bulk update
  updateSettings: (settings: Partial<SettingsState>) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      // Default values
      useDeepTraversal: true,
      maxDepth: 5,
      maxBranches: 3,
      domainTemplate: "general",
      language: "ko",
      
      // Individual setters
      setUseDeepTraversal: (value) => set({ useDeepTraversal: value }),
      setMaxDepth: (value) => set({ maxDepth: value }),
      setMaxBranches: (value) => set({ maxBranches: value }),
      setDomainTemplate: (value) => set({ domainTemplate: value }),
      setLanguage: (value) => set({ language: value }),
      
      // Bulk update
      updateSettings: (settings) => set(settings),
    }),
    {
      name: "treerag-settings",
    }
  )
);
