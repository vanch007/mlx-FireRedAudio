import React, { useState } from "react";
import { useApp } from "../../context/AppContext";
import { api } from "../../services/api";
import { Music, Mic2, Trash2, Play, Upload, Download, ArrowRight, Mic, MessageSquareQuote, Sliders } from "lucide-react";

export const AssetsVoicesView: React.FC = () => {
  const { assets, voices, playAudio, refreshData, sendVoiceToTTS, sendAssetToTool } = useApp();
  const [tab, setTab] = useState<"assets" | "voices">("assets");

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await api.uploadAsset(file);
      await refreshData();
    } catch (err: any) {
      alert("上传失败: " + err.message);
    }
  };

  const handleDeleteAsset = async (id: string) => {
    if (!confirm("确定要删除此音频素材吗？")) return;
    try {
      await api.deleteAsset(id);
      await refreshData();
    } catch (err: any) {
      alert("删除失败: " + err.message);
    }
  };

  const handleDeleteVoice = async (id: string) => {
    if (!confirm("确定要删除此声音配置吗？")) return;
    try {
      await api.deleteVoice(id);
      await refreshData();
    } catch (err: any) {
      alert("删除失败: " + err.message);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">素材库与预设声音库</h2>
          <p className="text-xs text-slate-400 mt-1">管理上传的原声素材、生成的音频结果以及保存的声音克隆音色模板</p>
        </div>
        <label className="cursor-pointer px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-md flex items-center space-x-2 transition">
          <Upload className="w-4 h-4" />
          <span>上传音频素材</span>
          <input type="file" accept="audio/*" onChange={handleUpload} className="hidden" />
        </label>
      </div>

      <div className="flex border-b border-slate-800 space-x-6 text-sm font-medium">
        <button
          onClick={() => setTab("assets")}
          className={"pb-3 flex items-center space-x-2 border-b-2 transition " + (tab === "assets" ? "border-sky-500 text-sky-400" : "border-transparent text-slate-400 hover:text-slate-200")}
        >
          <Music className="w-4 h-4" />
          <span>全部音频素材 ({assets.length})</span>
        </button>
        <button
          onClick={() => setTab("voices")}
          className={"pb-3 flex items-center space-x-2 border-b-2 transition " + (tab === "voices" ? "border-sky-500 text-sky-400" : "border-transparent text-slate-400 hover:text-slate-200")}
        >
          <Mic2 className="w-4 h-4" />
          <span>音色克隆模板库 ({voices.length})</span>
        </button>
      </div>

      {/* Tab 1: Assets */}
      {tab === "assets" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {assets.length === 0 ? (
            <div className="col-span-full py-16 text-center text-slate-500 text-xs">暂无音频素材，点击右上角上传</div>
          ) : (
            assets.map((a) => (
              <div key={a.id} className="glass-card rounded-2xl p-4 flex flex-col justify-between space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2 truncate max-w-[200px]">
                    <div className="p-2 rounded-xl bg-sky-500/10 text-sky-400">
                      <Music className="w-4 h-4" />
                    </div>
                    <div className="truncate">
                      <h4 className="font-bold text-slate-200 text-xs truncate" title={a.name}>{a.name}</h4>
                      <p className="text-[10px] text-slate-500">{a.source} · {a.duration}s · {a.sample_rate}Hz</p>
                    </div>
                  </div>
                  <button onClick={() => handleDeleteAsset(a.id)} className="text-slate-500 hover:text-rose-400 p-1">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-slate-800 text-xs">
                  <button
                    onClick={() => playAudio(a.media_url, a.name, a.duration)}
                    className="px-3 py-1.5 rounded-lg bg-sky-500/20 text-sky-300 hover:bg-sky-500/30 transition flex items-center space-x-1"
                  >
                    <Play className="w-3 h-3" />
                    <span>播放</span>
                  </button>
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => sendAssetToTool(a.id, 'asr')}
                      className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
                      title="识别/问答"
                    >
                      <Mic className="w-3.5 h-3.5 text-sky-400" />
                    </button>
                    <button
                      onClick={() => sendAssetToTool(a.id, 'tts')}
                      className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
                      title="声音克隆参考"
                    >
                      <MessageSquareQuote className="w-3.5 h-3.5 text-indigo-400" />
                    </button>
                    <button
                      onClick={() => sendAssetToTool(a.id, 'edit_acoustic')}
                      className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
                      title="语音编辑"
                    >
                      <Sliders className="w-3.5 h-3.5 text-amber-400" />
                    </button>
                    <a href={a.media_url} download={a.name} className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200">
                      <Download className="w-3.5 h-3.5" />
                    </a>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Tab 2: Voices */}
      {tab === "voices" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {voices.length === 0 ? (
            <div className="col-span-full py-16 text-center text-slate-500 text-xs">暂无音色模板，可在声音克隆工具中一键保存参考声音</div>
          ) : (
            voices.map((v) => {
              const asset = assets.find((a) => a.id === v.audio_asset_id);
              return (
                <div key={v.id} className="glass-card rounded-2xl p-5 space-y-4 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-indigo-600 text-white flex items-center justify-center font-bold text-xs">
                          {v.name.slice(0, 1)}
                        </div>
                        <div>
                          <h4 className="font-bold text-slate-200 text-sm">{v.name}</h4>
                          <span className="text-[10px] text-slate-500 uppercase">{v.language === "zh" ? "中文普通话" : "English"}</span>
                        </div>
                      </div>
                      <button onClick={() => handleDeleteVoice(v.id)} className="text-slate-500 hover:text-rose-400 p-1">
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300 italic">
                      "{v.prompt_text}"
                    </div>
                  </div>
                  <div className="flex items-center justify-between pt-2 border-t border-slate-800">
                    {asset && (
                      <button
                        onClick={() => playAudio(asset.media_url, v.name + " (参考音色)", asset.duration)}
                        className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1"
                      >
                        <Play className="w-3 h-3" />
                        <span>试听音色</span>
                      </button>
                    )}
                    <button
                      onClick={() => sendVoiceToTTS(v)}
                      className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold flex items-center space-x-1 shadow"
                    >
                      <span>使用此音色配音</span>
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
};
