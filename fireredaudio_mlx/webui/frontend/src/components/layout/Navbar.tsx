import React from "react";
import { useApp } from "../../context/AppContext";
import { Activity, Cpu, Trash2, ExternalLink } from "lucide-react";

export const Navbar: React.FC = () => {
  const { systemStatus, clearCache, jobs } = useApp();
  const activeJobs = jobs.filter((j) => j.status === "inferencing" || j.status === "loading" || j.status === "queued").length;

  const getStatusColor = (status?: string) => {
    switch (status) {
      case "ready": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "loading": return "bg-amber-500/20 text-amber-400 border-amber-500/30 animate-pulse";
      case "error": return "bg-rose-500/20 text-rose-400 border-rose-500/30";
      default: return "bg-slate-700/50 text-slate-400 border-slate-600/30";
    }
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40">
      <div className="flex items-center space-x-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20 font-bold text-white tracking-wider">
          FR
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="font-bold text-base tracking-tight text-white">FireRedAudio Studio</h1>
            <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">
              Apple Silicon MLX 9B
            </span>
          </div>
        </div>
      </div>
      <div className="flex items-center space-x-4 text-xs font-medium">
        <div className={"flex items-center space-x-2 px-3 py-1.5 rounded-full border " + getStatusColor(systemStatus?.status)}>
          <span className="w-2 h-2 rounded-full bg-current"></span>
          <span>{systemStatus?.status === "ready" ? ("模型就绪 (" + (systemStatus?.load_time_seconds || 0) + "s)") : (systemStatus?.status === "loading" ? "模型加载中..." : (systemStatus?.status === "error" ? "加载异常" : "未加载"))}</span>
        </div>
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/80 text-slate-300 border border-slate-700/50">
          <Cpu className="w-3.5 h-3.5 text-sky-400" />
          <span>Metal 显存: {systemStatus?.memory.active_memory_gb || 0} GB</span>
          {systemStatus?.memory.total_memory_gb && <span className="text-slate-500">/ {systemStatus.memory.total_memory_gb} GB</span>}
        </div>
        {activeJobs > 0 && (
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/30">
            <Activity className="w-3.5 h-3.5 animate-spin" />
            <span>{activeJobs} 个任务运行中</span>
          </div>
        )}
        <button onClick={clearCache} title="清理 Metal 显存碎片" className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-sky-400 hover:bg-slate-700 transition">
          <Trash2 className="w-4 h-4" />
        </button>
        <a href="/docs" target="_blank" rel="noreferrer" className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition">
          <span>API 文档</span>
          <ExternalLink className="w-3 h-3 text-slate-400" />
        </a>
      </div>
    </header>
  );
};
