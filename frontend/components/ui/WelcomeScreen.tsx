import { FileText } from "lucide-react";

interface WelcomeScreenProps {
  t: {
    welcomeTitle: string;
    welcomeDesc: string;
    shortcutKey: string;
    newSession: string;
  };
}

export default function WelcomeScreen({ t }: WelcomeScreenProps) {
  return (
    <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-80 pb-20">
      <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6 shadow-xl">
        <FileText className="w-10 h-10 text-white" />
      </div>
      <h2 className="text-2xl font-bold text-slate-700 mb-2">{t.welcomeTitle}</h2>
      <p className="max-w-md text-center text-slate-500">
        {t.welcomeDesc}
      </p>
      <p className="text-xs text-slate-400 mt-4">
        {t.shortcutKey}: <kbd className="px-2 py-1 bg-slate-100 rounded">Ctrl+K</kbd> {t.newSession}
      </p>
    </div>
  );
}
