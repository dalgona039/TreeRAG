import { Settings as SettingsIcon } from "lucide-react";

interface SettingsPanelProps {
  showSettings: boolean;
  domainTemplate: string;
  language: string;
  useDeepTraversal: boolean;
  maxDepth: number;
  maxBranches: number;
  onDomainChange: (value: string) => void;
  onLanguageChange: (value: string) => void;
  onDeepTraversalChange: (value: boolean) => void;
  onMaxDepthChange: (value: number) => void;
  onMaxBranchesChange: (value: number) => void;
  t: any;
}

export default function SettingsPanel({ 
  showSettings,
  domainTemplate,
  language,
  useDeepTraversal,
  maxDepth,
  maxBranches,
  onDomainChange,
  onLanguageChange,
  onDeepTraversalChange,
  onMaxDepthChange,
  onMaxBranchesChange,
  t 
}: SettingsPanelProps) {
  if (!showSettings) return null;

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-200 p-4">
      <div className="max-w-4xl mx-auto">
        <h3 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
          <SettingsIcon size={16} className="text-indigo-600" />
          {t.analysisSettings}
        </h3>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            📋 {t.documentDomain}
          </label>
          <select
            value={domainTemplate}
            onChange={(e) => onDomainChange(e.target.value)}
            className="w-full px-3 py-2 border border-blue-300 rounded-lg text-sm bg-white focus:ring-2 focus:ring-indigo-500"
          >
            <option value="general">📋 {t.general}</option>
            <option value="medical">🏥 {t.medical}</option>
            <option value="legal">⚖️ {t.legal}</option>
            <option value="financial">💼 {t.financial}</option>
            <option value="academic">🎓 {t.academic}</option>
          </select>
          <p className="text-xs text-slate-500 mt-1">
            {t.domainOptimized}
          </p>
        </div>
        
        <div className="mb-4">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            🌐 {t.responseLanguage}
          </label>
          <select
            value={language}
            onChange={(e) => onLanguageChange(e.target.value)}
            className="w-full px-3 py-2 border border-blue-300 rounded-lg text-sm bg-white focus:ring-2 focus:ring-indigo-500"
          >
            <option value="ko">🇰🇷 {t.korean}</option>
            <option value="en">🇺🇸 {t.english}</option>
            <option value="ja">🇯🇵 {t.japanese}</option>
          </select>
          <p className="text-xs text-slate-500 mt-1">
            {t.languageOptimized}
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useDeepTraversal}
                onChange={(e) => onDeepTraversalChange(e.target.checked)}
                className="w-4 h-4 text-indigo-600 rounded"
              />
              <span className="text-sm font-medium text-slate-700">{t.deepTraversal}</span>
            </label>
            <p className="text-xs text-slate-500 mt-1 ml-6">
              {useDeepTraversal ? t.deepTraversalDesc : t.flatModeDesc}
            </p>
          </div>

          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <label className="block text-sm font-medium text-slate-700 mb-1">
              {t.maxDepth}
            </label>
            <input
              type="number"
              value={maxDepth}
              onChange={(e) => onMaxDepthChange(Number(e.target.value))}
              min="1"
              max="10"
              disabled={!useDeepTraversal}
              className="w-full px-3 py-1 border border-slate-300 rounded text-sm disabled:bg-slate-100 disabled:text-slate-400"
            />
            <p className="text-xs text-slate-500 mt-1">
              {t.maxDepthDesc}
            </p>
          </div>

          <div className="bg-white p-3 rounded-lg border border-blue-200">
            <label className="block text-sm font-medium text-slate-700 mb-1">
              {t.maxBranches}
            </label>
            <input
              type="number"
              value={maxBranches}
              onChange={(e) => onMaxBranchesChange(Number(e.target.value))}
              min="1"
              max="10"
              disabled={!useDeepTraversal}
              className="w-full px-3 py-1 border border-slate-300 rounded text-sm disabled:bg-slate-100 disabled:text-slate-400"
            />
            <p className="text-xs text-slate-500 mt-1">
              {t.maxBranchesDesc}
            </p>
          </div>
        </div>
        <div className="mt-3 text-xs text-blue-700 bg-blue-100 p-2 rounded">
          💡 <strong>{t.tip}:</strong> {t.tipMessage}
        </div>
      </div>
    </div>
  );
}
