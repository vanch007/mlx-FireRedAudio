import React, { useState } from "react";
import { useApp } from "../../context/AppContext";
import { api } from "../../services/api";
import { ListOrdered, CheckCircle2, AlertCircle, Clock, Play, Ban, RotateCcw, AlertTriangle } from "lucide-react";

export const JobsView: React.FC = () => {
  const { jobs, playAudio, refreshData } = useApp();
  const [filter, setFilter] = useState<string>("all");

  const filteredJobs = jobs.filter((j) => {
    if (filter === "running") return j.status === "loading" || j.status === "preprocessing" || j.status === "inferencing" || j.status === "queued";
    if (filter === "completed") return j.status === "completed";
    if (filter === "failed") return j.status === "failed" || j.status === "interrupted";
    return true;
  });

  const handleCancel = async (id: string) => {
    try {
      await api.cancelJob(id);
      await refreshData();
    } catch (err: any) {
      alert("取消失败: " + err.message);
    }
  };

  const handleRetry = async (id: string) => {
    try {
      await api.retryJob(id);
      await refreshData();
    } catch (err: any) {
      alert("重试失败: " + err.message);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[11px] font-semibold flex items-center space-x-1"><CheckCircle2 className="w-3 h-3" /><span>已完成</span></span>;
      case "failed":
        return <span className="px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 text-[11px] font-semibold flex items-center space-x-1"><AlertCircle className="w-3 h-3" /><span>失败</span></span>;
      case "interrupted":
        return <span className="px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 text-[11px] font-semibold flex items-center space-x-1"><AlertTriangle className="w-3 h-3" /><span>已中断</span></span>;
      case "cancelled":
        return <span className="px-2.5 py-1 rounded-full bg-slate-700/50 text-slate-400 text-[11px] font-semibold">已取消</span>;
      default:
        return <span className="px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 text-[11px] font-semibold flex items-center space-x-1 animate-pulse"><Clock className="w-3 h-3" /><span>处理中</span></span>;
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">任务调度中心</h2>
          <p className="text-xs text-slate-400 mt-1">实时查看单实例排队队列、Metal 推理耗时、生成状态与输出历史</p>
        </div>
        <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs font-medium">
          {["all", "running", "completed", "failed"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={"px-3 py-1.5 rounded-lg transition " + (filter === f ? "bg-sky-500 text-white shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200")}
            >
              {f === "all" ? "全部" : f === "running" ? "运行/排队中" : f === "completed" ? "已完成" : "失败"}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-card rounded-2xl overflow-hidden border border-slate-800">
        {filteredJobs.length === 0 ? (
          <div className="py-24 text-center text-slate-500 text-xs">暂无符合条件的任务记录</div>
        ) : (
          <div className="divide-y divide-slate-800/80">
            {filteredJobs.map((j) => (
              <div key={j.id} className="p-5 flex items-center justify-between hover:bg-slate-900/40 transition">
                <div className="space-y-1.5 max-w-xl">
                  <div className="flex items-center space-x-3">
                    <span className="font-bold text-sm text-white uppercase tracking-wider">{j.task}</span>
                    <span className="font-mono text-[11px] text-slate-500">#{j.id}</span>
                    {getStatusBadge(j.status)}
                  </div>
                  <p className="text-xs text-slate-400">{j.current_step}</p>
                  {j.error && <p className="text-[11px] text-rose-400 font-mono bg-rose-950/30 p-2 rounded-lg">{j.error}</p>}
                  {j.result?.transcript && <p className="text-xs text-slate-200 bg-slate-950/60 p-2 rounded-lg font-medium">{j.result.transcript}</p>}
                  {j.result?.answer && <p className="text-xs text-slate-200 bg-slate-950/60 p-2 rounded-lg font-medium">{j.result.answer}</p>}
                </div>

                <div className="flex items-center space-x-4 text-xs">
                  <div className="text-right">
                    <p className="font-mono text-slate-300 font-semibold">{j.latency_seconds ? (j.latency_seconds + "s" + (j.result?.rtf ? " (RTF " + j.result.rtf + ")" : "")) : "--"}</p>
                    <p className="text-[10px] text-slate-500">{new Date(j.created_at * 1000).toLocaleTimeString()}</p>
                  </div>
                  {j.result?.media_url && (
                    <button
                      onClick={() => playAudio(j.result?.media_url, "Result-" + j.id, j.result?.duration_s)}
                      className="p-2.5 rounded-xl bg-sky-500/20 text-sky-300 hover:bg-sky-500/30 transition flex items-center space-x-1"
                    >
                      <Play className="w-4 h-4" />
                      <span>播放</span>
                    </button>
                  )}
                  {(j.status === "failed" || j.status === "interrupted" || j.status === "cancelled") && (
                    <button
                      onClick={() => handleRetry(j.id)}
                      className="p-2 rounded-lg bg-slate-800 hover:bg-sky-500/20 hover:text-sky-300 text-slate-400 transition"
                      title="重新排队重试此任务"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  )}
                  {j.status === "queued" && (
                    <button
                      onClick={() => handleCancel(j.id)}
                      className="p-2 rounded-lg bg-slate-800 hover:bg-rose-500/20 hover:text-rose-400 text-slate-400 transition"
                      title="取消排队任务"
                    >
                      <Ban className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
