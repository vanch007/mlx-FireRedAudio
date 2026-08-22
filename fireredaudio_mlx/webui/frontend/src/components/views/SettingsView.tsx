import React, { useState } from "react";
import { useApp } from "../../context/AppContext";
import { api } from "../../services/api";
import { Cpu, HardDrive, Sparkles, Trash2, RefreshCw, CheckCircle2 } from "lucide-react";

export const SettingsView: React.FC = () => {
  const { systemStatus, presets, clearCache, refreshData } = useApp();
  const [modelPathInput, setModelPathInput] = useState<string>(systemStatus?.model_path || "models/FireRedAudio");
  const [loading, setLoading] = useState<boolean>(false);
  const [msg, setMsg] = useState<string>("");

  const handleReload = async () => {
    setLoading(true);
    try {
      const res = await api.loadModel(modelPathInput);
      setMsg("模型加载请求已发送！");
      await refreshData();
    } catch (err: any) {
      alert("加载失败: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8 text-xs">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">硬件监控与引擎设置</h2>
        <p className="text-slate-400 mt-1">查看 Apple Silicon 统一内存运行指标，管理模型权重目录与默认推理预设</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Hardware Status Card */}
        <div className="glass-card rounded-2xl p-6 space-y-4">
          <h3 className="font-bold text-slate-200 text-sm flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-sky-400" />
            <span>Apple Silicon Metal 状态</span>
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">设备型号</span>
              <span className="font-semibold text-slate-200">{systemStatus?.memory.device_name || "Apple GPU"}</span>
            </div>
            <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">MLX 框架版本</span>
              <span className="font-mono text-slate-200">{systemStatus?.memory.mlx_version || "0.32.1"}</span>
            </div>
            <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Metal 当前活跃显存</span>
              <span className="font-semibold text-sky-400">{(systemStatus?.memory.active_memory_gb || 0) + " GB"}</span>
            </div>
            <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">显存使用峰值 (Peak)</span>
              <span className="font-semibold text-slate-300">{(systemStatus?.memory.peak_memory_gb || 0) + " GB"}</span>
            </div>
            <div className="flex justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">显存缓存池 (Cache Pool)</span>
              <span className="font-semibold text-slate-400">{(systemStatus?.memory.cache_memory_gb || 0) + " GB"}</span>
            </div>
          </div>
          <button
            onClick={clearCache}
            className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold flex items-center justify-center space-x-2 transition"
          >
            <Trash2 className="w-4 h-4 text-sky-400" />
            <span>立即清理 Metal 显存缓存碎片</span>
          </button>
        </div>

        {/* Model Weights Setup */}
        <div className="glass-card rounded-2xl p-6 space-y-4">
          <h3 className="font-bold text-slate-200 text-sm flex items-center space-x-2">
            <HardDrive className="w-4 h-4 text-sky-400" />
            <span>模型权重与单例生命周期</span>
          </h3>
          <div className="space-y-2">
            <label className="text-slate-300 font-semibold">FireRedAudio 本地权重目录</label>
            <div className="flex flex-wrap gap-1.5 mb-2">
              {[
                { path: "models/FireRedAudio", label: "BF16 原始高精度 (20.5GB)" },
                { path: "models/FireRedAudio-8bit", label: "⚡ 8-bit 实时加速 (13.5GB 推荐)" },
                { path: "models/FireRedAudio-4bit", label: "🚀 4-bit 极速轻量 (9.8GB)" },
              ].map((m) => (
                <button
                  key={m.path}
                  onClick={() => setModelPathInput(m.path)}
                  className={"px-2.5 py-1 rounded-lg text-[11px] border transition " + (modelPathInput === m.path ? "bg-sky-500/20 text-sky-300 border-sky-500/40 font-semibold" : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200")}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <input
              type="text"
              value={modelPathInput}
              onChange={(e) => setModelPathInput(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200 font-mono text-xs"
            />
            <p className="text-[11px] text-slate-500">支持原生 BF16、8-bit 以及 4-bit 独立量化模型权重目录</p>
          </div>
          <div className="flex items-center space-x-3 pt-2">
            <button
              disabled={loading}
              onClick={handleReload}
              className="px-4 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-white font-semibold flex items-center space-x-2 transition"
            >
              <RefreshCw className={"w-4 h-4 " + (loading ? "animate-spin" : "")} />
              <span>重新加载模型</span>
            </button>
            {msg && <span className="text-emerald-400 flex items-center space-x-1"><CheckCircle2 className="w-3.5 h-3.5" /><span>{msg}</span></span>}
          </div>
        </div>
      </div>

      {/* Quality Presets Table */}
      <div className="glass-card rounded-2xl p-6 space-y-4">
        <h3 className="font-bold text-slate-200 text-sm flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-sky-400" />
          <span>已内置的性能与质量预设配置</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(presets).map(([k, v]) => (
            <div key={k} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-sm text-sky-300">{v.name}</span>
                <span className="font-mono text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">Flow: {v.n_timesteps} steps</span>
              </div>
              <p className="text-slate-400 text-[11px] leading-relaxed">{v.description}</p>
              <div className="text-[10px] text-slate-500 font-mono pt-1">
                ASR Beams: {v.asr_beam_size} · Temp: {v.temperature} · TopK: {v.top_k}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
