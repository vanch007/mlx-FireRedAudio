import React from "react";
import { useApp } from "../../context/AppContext";
import { LayoutDashboard, Wand2, FolderKanban, Music2, ListOrdered, Settings, Mic, MessageSquareQuote, Sliders, Sparkles } from "lucide-react";

export const Sidebar: React.FC = () => {
  const { currentTab, setCurrentTab, toolSubTab, setToolSubTab, jobs } = useApp();
  const activeJobsCount = jobs.filter((j) => j.status === "queued" || j.status === "loading" || j.status === "inferencing").length;

  const navItems = [
    { id: "overview", label: "工作台总览", icon: LayoutDashboard },
    { id: "tools", label: "快捷语音工具", icon: Wand2 },
    { id: "projects", label: "项目工作区", icon: FolderKanban },
    { id: "assets_voices", label: "素材与声音库", icon: Music2 },
    { id: "jobs", label: "任务调度队列", icon: ListOrdered, badge: activeJobsCount > 0 ? activeJobsCount : undefined },
    { id: "settings", label: "硬件与设置", icon: Settings },
  ];

  const toolSubItems = [
    { id: "asr_qa", label: "转写与问答 (ASR/QA)", icon: Mic },
    { id: "tts", label: "克隆与配音 (TTS)", icon: MessageSquareQuote },
    { id: "edit", label: "声学与语义编辑", icon: Sliders },
    { id: "voice_design", label: "音色设计 (Voice Design)", icon: Sparkles },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-900/50 flex flex-col justify-between p-4 shrink-0">
      <div className="space-y-6">
        <div className="space-y-1">
          <p className="px-3 text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">主导航</p>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentTab(item.id)}
                className={"w-full flex items-center justify-between px-3 py-2.5 rounded-xl font-medium text-sm transition-all " + (isActive ? "bg-sky-500/10 text-sky-400 border border-sky-500/20 shadow-sm" : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60")}
              >
                <div className="flex items-center space-x-3">
                  <Icon className={"w-4 h-4 " + (isActive ? "text-sky-400" : "text-slate-400")} />
                  <span>{item.label}</span>
                </div>
                {item.badge !== undefined && (
                  <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {currentTab === "tools" && (
          <div className="space-y-1 pt-2 border-t border-slate-800/80">
            <p className="px-3 text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">工具箱分类</p>
            {toolSubItems.map((sub) => {
              const SubIcon = sub.icon;
              const isSubActive = toolSubTab === sub.id;
              return (
                <button
                  key={sub.id}
                  onClick={() => setToolSubTab(sub.id)}
                  className={"w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-xs font-medium transition " + (isSubActive ? "bg-slate-800 text-sky-300 font-semibold" : "text-slate-400 hover:text-slate-300 hover:bg-slate-800/40")}
                >
                  <SubIcon className={"w-3.5 h-3.5 " + (isSubActive ? "text-sky-400" : "text-slate-500")} />
                  <span>{sub.label}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60 text-[11px] text-slate-500 space-y-1">
        <div className="flex items-center justify-between font-semibold text-slate-400">
          <span>M3 Max Native</span>
          <span className="text-emerald-400">零拷贝</span>
        </div>
        <p>1,741 权重无缝挂载 · 统一内存 Metal 原生加速</p>
      </div>
    </aside>
  );
};
