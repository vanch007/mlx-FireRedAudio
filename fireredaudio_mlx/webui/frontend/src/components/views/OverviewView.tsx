import React from "react";
import { useApp } from "../../context/AppContext";
import { Mic, MessageSquareQuote, Sliders, Sparkles, Clock, Cpu, Play, ArrowRight } from "lucide-react";

export const OverviewView: React.FC = () => {
  const { setCurrentTab, setToolSubTab, systemStatus, assets, projects, jobs, playAudio } = useApp();

  const tools = [
    {
      id: "asr_qa",
      title: "转写与智能问答",
      desc: "支持高准确率多语言 ASR、长音频理解与 Chain-of-Thought 深度推理",
      icon: Mic,
      color: "from-blue-500 to-sky-600",
    },
    {
      id: "tts",
      title: "声音克隆与配音",
      desc: "零样本 Zero-shot Timbre Continuation，从短音频克隆音色并生成目标语音",
      icon: MessageSquareQuote,
      color: "from-indigo-500 to-purple-600",
    },
    {
      id: "edit",
      title: "声学与语义编辑",
      desc: "无损调节语速 (0.5x~2.0x)、音高与音量，或通过自然语言删除/重写语音内容",
      icon: Sliders,
      color: "from-amber-500 to-orange-600",
    },
    {
      id: "voice_design",
      title: "自然语言音色设计",
      desc: "通过自然语言描述（如“温柔知性女主播”）直接生成全新声线及语音",
      icon: Sparkles,
      color: "from-rose-500 to-pink-600",
    },
  ];

  const openTool = (toolId: string) => {
    setToolSubTab(toolId);
    setCurrentTab("tools");
  };

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-slate-700/60 p-8 shadow-2xl">
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Apple Silicon M3 Max 全能力本地部署</span>
          </div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
            FireRedAudio Studio — 本地音频多模态工作流中心
          </h2>
          <p className="text-slate-300 text-sm leading-relaxed">
            基于 Apple MLX 原生框架构建，统一整合 9B 参数解耦架构（ASR 理解、RedAE 声码器、Flow DiT 生成器与 Qwen3.5 骨干）。零云端依赖，数据完全本地私有。
          </p>
          <div className="pt-2 flex items-center space-x-4">
            <button
              onClick={() => openTool("tts")}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white font-semibold text-sm shadow-lg shadow-sky-500/25 transition"
            >
              <span>开始声音克隆</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => openTool("asr_qa")}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-sm border border-slate-700 transition"
            >
              <span>语音识别 / 问答</span>
            </button>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center space-x-2">
          <span>四大核心能力工具箱</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {tools.map((t) => {
            const Icon = t.icon;
            return (
              <div
                key={t.id}
                onClick={() => openTool(t.id)}
                className="group cursor-pointer glass-card rounded-2xl p-5 hover:border-sky-500/40 transition-all duration-200 hover:-translate-y-1 shadow-lg"
              >
                <div className={"w-10 h-10 rounded-xl bg-gradient-to-tr " + t.color + " flex items-center justify-center text-white mb-4 shadow-md"}>
                  <Icon className="w-5 h-5" />
                </div>
                <h4 className="font-bold text-slate-100 text-base mb-1 group-hover:text-sky-400 transition">{t.title}</h4>
                <p className="text-slate-400 text-xs leading-relaxed">{t.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-card rounded-2xl p-6 space-y-4">
          <h3 className="font-bold text-slate-200 text-sm flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-sky-400" />
            <span>本地 Apple Silicon 硬件状态</span>
          </h3>
          <div className="space-y-3 text-xs">
            <div className="flex justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-400">计算架构</span>
              <span className="font-semibold text-slate-200">{systemStatus?.memory.device_name || "Apple GPU (Metal)"}</span>
            </div>
            <div className="flex justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-400">MLX 当前显存</span>
              <span className="font-semibold text-sky-400">{(systemStatus?.memory.active_memory_gb || 0) + " GB"}</span>
            </div>
            <div className="flex justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-400">项目 / 素材总数</span>
              <span className="font-semibold text-slate-200">{projects.length + " 个项目 / " + assets.length + " 条音频"}</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 glass-card rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-slate-200 text-sm flex items-center space-x-2">
              <Clock className="w-4 h-4 text-sky-400" />
              <span>最近推理任务</span>
            </h3>
            <button onClick={() => setCurrentTab("jobs")} className="text-xs text-sky-400 hover:underline">
              {"查看全部 (" + jobs.length + ")"}
            </button>
          </div>

          {jobs.length === 0 ? (
            <div className="py-8 text-center text-slate-500 text-xs">暂无历史任务，点击上方工具立即体验</div>
          ) : (
            <div className="space-y-2">
              {jobs.slice(0, 4).map((job) => (
                <div key={job.id} className="flex items-center justify-between p-3 rounded-xl bg-slate-900/40 border border-slate-800/80 text-xs">
                  <div className="flex items-center space-x-3">
                    <span className={"w-2 h-2 rounded-full " + (job.status === "completed" ? "bg-emerald-400" : job.status === "failed" ? "bg-rose-400" : "bg-amber-400 animate-pulse")}></span>
                    <div>
                      <span className="font-bold text-slate-200 uppercase tracking-wider">{job.task}</span>
                      <p className="text-slate-400 text-[11px] truncate max-w-xs">{job.current_step}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="text-slate-500 text-[11px] font-mono">{job.latency_seconds ? (job.latency_seconds + "s" + (job.result?.rtf ? " · RTF " + job.result.rtf : "")) : ""}</span>
                    {job.result?.media_url && (
                      <button
                        onClick={() => playAudio(job.result?.media_url, "Job-" + job.id, job.result?.duration_s)}
                        className="p-1.5 rounded-lg bg-sky-500/20 text-sky-300 hover:bg-sky-500/30 transition"
                      >
                        <Play className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
