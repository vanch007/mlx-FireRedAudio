import React, { useState, useEffect } from "react";
import { useApp } from "../../context/AppContext";
import { api } from "../../services/api";
import { Mic, MessageSquareQuote, Sliders, Sparkles, Upload, Play, Loader2, Copy, Check, ChevronDown, ChevronUp, Save, Settings2, X } from "lucide-react";

export const QuickToolsView: React.FC = () => {
  const { toolSubTab, setToolSubTab, assets, voices, presets, playAudio, refreshData, activeProjectId, toolPrefill } = useApp();
  const [selectedPreset, setSelectedPreset] = useState<string>("balanced");
  const [submitting, setSubmitting] = useState<boolean>(false)
  const [jobResult, setJobResult] = useState<any>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [showThinking, setShowThinking] = useState<boolean>(true);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // ASR & QA State
  const [asrAudioIds, setAsrAudioIds] = useState<string[]>([]);
  const [asrTask, setAsrTask] = useState<"asr" | "understand">("asr");
  const [asrPrompt, setAsrPrompt] = useState<string>("Transcribe speech to text.");
  const [enableThinking, setEnableThinking] = useState<boolean>(false);

  // TTS State
  const [ttsPromptAudioId, setTtsPromptAudioId] = useState<string>("");
  const [ttsPromptText, setTtsPromptText] = useState<string>("");
  const [ttsTargetText, setTtsTargetText] = useState<string>("你好，欢迎使用 FireRedAudio MLX 语音工作台！");
  const [ttsLanguage, setTtsLanguage] = useState<string>("zh");
  const [saveVoiceName, setSaveVoiceName] = useState<string>("");

  // Speech Edit State
  const [editAudioId, setEditAudioId] = useState<string>("");
  const [editType, setEditType] = useState<"acoustic" | "semantic">("acoustic");
  const [acousticMode, setAcousticMode] = useState<"speed" | "pitch" | "volume" | "custom">("speed");
  const [editSpeed, setEditSpeed] = useState<number>(1.0);
  const [editPitch, setEditPitch] = useState<number>(0);
  const [editVolume, setEditVolume] = useState<number>(1.0);
  const [customAcousticInst, setCustomAcousticInst] = useState<string>("adjust the speed to 0.8");
  const [editInstruction, setEditInstruction] = useState<string>("delete '比普通的茶叶要'");

  // Voice Design State
  const [vdInstruction, setVdInstruction] = useState<string>("温柔清晰的广播女声，语速平稳，音色清亮甜美");
  const [vdText, setVdText] = useState<string>("欢迎收听今日新闻，我们将为您带来最新的科技前沿报道。");

  // Advanced Params
  const [advMaxTextTokens, setAdvMaxTextTokens] = useState<number>(512);
  const [advMaxAudioSteps, setAdvMaxAudioSteps] = useState<number>(750);
  const [advCfg, setAdvCfg] = useState<number>(2.0);

  useEffect(() => {
    if (toolPrefill.asrAudioIds && toolPrefill.asrAudioIds.length > 0) {
      setAsrAudioIds(toolPrefill.asrAudioIds);
      if (toolPrefill.asrTask) setAsrTask(toolPrefill.asrTask);
    }
    if (toolPrefill.ttsPromptAudioId) setTtsPromptAudioId(toolPrefill.ttsPromptAudioId);
    if (toolPrefill.ttsPromptText) setTtsPromptText(toolPrefill.ttsPromptText);
    if (toolPrefill.ttsLanguage) setTtsLanguage(toolPrefill.ttsLanguage);
    if (toolPrefill.editAudioId) setEditAudioId(toolPrefill.editAudioId);
    if (toolPrefill.editType) setEditType(toolPrefill.editType);
  }, [toolPrefill]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, target: "asr" | "tts" | "edit") => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const asset = await api.uploadAsset(file, activeProjectId || undefined);
      await refreshData();
      if (target === "asr") setAsrAudioIds((prev) => [...prev, asset.id]);
      if (target === "tts") setTtsPromptAudioId(asset.id);
      if (target === "edit") setEditAudioId(asset.id);
    } catch (err: any) {
      alert("上传失败: " + err.message);
    }
  };

  const applyVoiceTemplate = (v: any) => {
    setTtsPromptAudioId(v.audio_asset_id);
    setTtsPromptText(v.prompt_text);
    setTtsLanguage(v.language);
  };

  const handleSaveVoice = async () => {
    if (!saveVoiceName || !ttsPromptAudioId || !ttsPromptText) {
      alert("请填写声音名称、参考音频与参考文本");
      return;
    }
    try {
      await api.createVoice({
        name: saveVoiceName,
        prompt_text: ttsPromptText,
        audio_asset_id: ttsPromptAudioId,
        language: ttsLanguage,
      });
      setSaveVoiceName("");
      await refreshData();
      alert("声音模板已保存到声音库！");
    } catch (err: any) {
      alert("保存失败: " + err.message);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    setJobResult(null);
    const presetCfg = presets[selectedPreset] || { n_timesteps: 4, inference_cfg: 2.0 };

    try {
      let job;
      if (toolSubTab === "asr_qa") {
        if (asrAudioIds.length === 0) throw new Error("请先选择或上传至少一个待识别音频");
        job = await api.createJob(asrTask, {
          audio_asset_ids: asrAudioIds,
          prompt: asrPrompt,
          enable_thinking: enableThinking,
          max_new_tokens: showAdvanced ? advMaxTextTokens : undefined,
        }, activeProjectId || undefined, selectedPreset);
      } else if (toolSubTab === "tts") {
        if (!ttsPromptAudioId || !ttsPromptText || !ttsTargetText) {
          throw new Error("请完整提供参考音频、参考文本及目标合成文本");
        }
        job = await api.createJob("tts", {
          prompt_audio_asset_id: ttsPromptAudioId,
          prompt_text: ttsPromptText,
          target_text: ttsTargetText,
          language: ttsLanguage,
          n_timesteps: presetCfg.n_timesteps,
          inference_cfg: showAdvanced ? advCfg : presetCfg.inference_cfg,
          max_new_audio_steps: showAdvanced ? advMaxAudioSteps : undefined,
        }, activeProjectId || undefined, selectedPreset);
      } else if (toolSubTab === "edit") {
        if (!editAudioId) throw new Error("请先选择或上传待编辑原音频");
        if (editType === "acoustic") {
          job = await api.createJob("edit_acoustic", {
            audio_asset_id: editAudioId,
            mode: acousticMode,
            speed: editSpeed,
            pitch: editPitch,
            volume: editVolume,
            instruction: acousticMode === "custom" ? customAcousticInst : undefined,
            n_timesteps: presetCfg.n_timesteps,
            inference_cfg: showAdvanced ? advCfg : presetCfg.inference_cfg,
            max_new_audio_steps: showAdvanced ? advMaxAudioSteps : undefined,
          }, activeProjectId || undefined, selectedPreset);
        } else {
          job = await api.createJob("edit_semantic", {
            audio_asset_id: editAudioId,
            instruction: editInstruction,
            n_timesteps: presetCfg.n_timesteps,
            inference_cfg: showAdvanced ? advCfg : presetCfg.inference_cfg,
            max_new_audio_steps: showAdvanced ? advMaxAudioSteps : undefined,
            max_new_text_tokens: showAdvanced ? advMaxTextTokens : undefined,
          }, activeProjectId || undefined, selectedPreset);
        }
      } else if (toolSubTab === "voice_design") {
        if (!vdInstruction || !vdText) throw new Error("请提供音色描述指令与目标文本");
        job = await api.createJob("voice_design", {
          instruction: vdInstruction,
          text: vdText,
          n_timesteps: presetCfg.n_timesteps,
          inference_cfg: showAdvanced ? advCfg : presetCfg.inference_cfg,
          max_new_audio_steps: showAdvanced ? advMaxAudioSteps : undefined,
          max_new_text_tokens: showAdvanced ? advMaxTextTokens : undefined,
        }, activeProjectId || undefined, selectedPreset);
      }

      if (job) {
        setActiveJobId(job.id);
        // Poll until finished
        const interval = setInterval(async () => {
          try {
            const updated = await api.listJobs();
            const current = updated.find((j) => j.id === job.id);
            if (current && (current.status === "completed" || current.status === "failed" || current.status === "cancelled")) {
              clearInterval(interval);
              setSubmitting(false);
              setJobResult(current);
              await refreshData();
            }
          } catch {}
        }, 1000);
      }
    } catch (err: any) {
      setSubmitting(false);
      alert("提交任务失败: " + err.message);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">快捷语音能力工作台</h2>
          <p className="text-xs text-slate-400 mt-1">选择下方功能模块，直接使用本地 Apple Metal 原生 9B 大模型进行推理</p>
        </div>
        <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 p-1 rounded-xl text-xs font-medium">
          {Object.entries(presets).map(([key, val]) => (
            <button
              key={key}
              onClick={() => setSelectedPreset(key)}
              className={"px-3 py-1.5 rounded-lg transition " + (selectedPreset === key ? "bg-sky-500 text-white shadow-sm font-semibold" : "text-slate-400 hover:text-slate-200")}
            >
              {val.name.split(" ")[0]}
            </button>
          ))}
        </div>
      </div>

      <div className="flex border-b border-slate-800 space-x-6 text-sm font-medium">
        <button
          onClick={() => { setToolSubTab("asr_qa"); setJobResult(null); }}
          className={"pb-3 flex items-center space-x-2 border-b-2 transition " + (toolSubTab === "asr_qa" ? "border-sky-500 text-sky-400" : "border-transparent text-slate-400 hover:text-slate-200")}
        >
          <Mic className="w-4 h-4" />
          <span>转写与智能问答 (ASR/QA)</span>
        </button>
        <button
          onClick={() => { setToolSubTab("tts"); setJobResult(null); }}
          className={"pb-3 flex items-center space-x-2 border-b-2 transition " + (toolSubTab === "tts" ? "border-sky-500 text-sky-400" : "border-transparent text-slate-400 hover:text-slate-200")}
        >
          <MessageSquareQuote className="w-4 h-4" />
          <span>声音克隆与配音 (TTS)</span>
        </button>
        <button
          onClick={() => { setToolSubTab("edit"); setJobResult(null); }}
          className={"pb-3 flex items-center space-x-2 border-b-2 transition " + (toolSubTab === "edit" ? "border-sky-500 text-sky-400" : "border-transparent text-slate-400 hover:text-slate-200")}
        >
          <Sliders className="w-4 h-4" />
          <span>声学与语义编辑</span>
        </button>
        <button
          onClick={() => { setToolSubTab("voice_design"); setJobResult(null); }}
          className={"pb-3 flex items-center space-x-2 border-b-2 transition " + (toolSubTab === "voice_design" ? "border-sky-500 text-sky-400" : "border-transparent text-slate-400 hover:text-slate-200")}
        >
          <Sparkles className="w-4 h-4" />
          <span>音色设计 (Voice Design)</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <div className="glass-card rounded-2xl p-6 space-y-6">
          {/* TAB 1: ASR & QA */}
          {toolSubTab === "asr_qa" && (
            <div className="space-y-4 text-xs">
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => { setAsrTask("asr"); setAsrPrompt("Transcribe speech to text."); setEnableThinking(false); }}
                  className={"flex-1 py-2 rounded-xl border text-center font-semibold transition " + (asrTask === "asr" ? "bg-sky-500/10 text-sky-400 border-sky-500/30" : "border-slate-800 text-slate-400 hover:bg-slate-800/50")}
                >
                  语音识别 (ASR 转写)
                </button>
                <button
                  onClick={() => { setAsrTask("understand"); setAsrPrompt("这个音频中有几个说话人？"); setEnableThinking(true); }}
                  className={"flex-1 py-2 rounded-xl border text-center font-semibold transition " + (asrTask === "understand" ? "bg-sky-500/10 text-sky-400 border-sky-500/30" : "border-slate-800 text-slate-400 hover:bg-slate-800/50")}
                >
                  音频理解问答 (QA/CoT)
                </button>
              </div>

              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">选择音频素材 (支持添加单音频或多音频)</label>
                {asrAudioIds.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mb-2">
                    {asrAudioIds.map((aid) => {
                      const a = assets.find((x) => x.id === aid);
                      return (
                        <span key={aid} className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-sky-500/15 text-sky-300 border border-sky-500/30">
                          <span>{a ? a.name : aid}</span>
                          <button onClick={() => setAsrAudioIds((prev) => prev.filter((x) => x !== aid))} className="text-slate-400 hover:text-rose-300">
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      );
                    })}
                  </div>
                )}
                <select
                  value=""
                  onChange={(e) => {
                    if (e.target.value && !asrAudioIds.includes(e.target.value)) {
                      setAsrAudioIds((prev) => [...prev, e.target.value]);
                    }
                  }}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  <option value="">+ 从素材库添加音频...</option>
                  {assets.map((a) => (
                    <option key={a.id} value={a.id}>{a.name} ({a.duration}s)</option>
                  ))}
                </select>
                <div className="flex items-center justify-between pt-1">
                  <label className="cursor-pointer inline-flex items-center space-x-1 text-sky-400 hover:underline">
                    <Upload className="w-3.5 h-3.5" />
                    <span>直接上传新音频</span>
                    <input type="file" accept="audio/*" onChange={(e) => handleFileUpload(e, "asr")} className="hidden" />
                  </label>
                  {asrAudioIds.length > 0 && (
                    <button
                      onClick={() => {
                        const a = assets.find((x) => x.id === asrAudioIds[0]);
                        if (a) playAudio(a.media_url, a.name, a.duration);
                      }}
                      className="text-slate-400 hover:text-slate-200 inline-flex items-center space-x-1"
                    >
                      <Play className="w-3.5 h-3.5" />
                      <span>试听首选音频</span>
                    </button>
                  )}
                </div>
              </div>

              {asrTask === "understand" && (
                <>
                  <div className="space-y-1.5">
                    <label className="text-slate-300 font-semibold">问答提示词 / 问题</label>
                    <textarea
                      rows={3}
                      value={asrPrompt}
                      onChange={(e) => setAsrPrompt(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200 focus:outline-none focus:border-sky-500"
                      placeholder="输入关于音频内容、说话人、情绪或背景声音的提问..."
                    />
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                    <div>
                      <p className="font-semibold text-slate-200">启用 Chain-of-Thought (思维链推理)</p>
                      <p className="text-[11px] text-slate-400">输出完整分析思路与证据核对过程</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={enableThinking}
                      onChange={(e) => setEnableThinking(e.target.checked)}
                      className="w-4 h-4 accent-sky-500 rounded cursor-pointer"
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {/* TAB 2: Zero-shot TTS */}
          {toolSubTab === "tts" && (
            <div className="space-y-4 text-xs">
              {voices.length > 0 && (
                <div className="space-y-1.5">
                  <label className="text-slate-300 font-semibold">快捷套用已保存的声音模板</label>
                  <div className="flex flex-wrap gap-2">
                    {voices.map((v) => (
                      <button
                        key={v.id}
                        onClick={() => applyVoiceTemplate(v)}
                        className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition"
                      >
                        {v.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">参考音色音频</label>
                <select
                  value={ttsPromptAudioId}
                  onChange={(e) => setTtsPromptAudioId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  <option value="">-- 选择参考音频素材 (建议 3~10 秒清晰语音) --</option>
                  {assets.map((a) => (
                    <option key={a.id} value={a.id}>{a.name} ({a.duration}s)</option>
                  ))}
                </select>
                <label className="cursor-pointer inline-flex items-center space-x-1 text-sky-400 hover:underline pt-1">
                  <Upload className="w-3.5 h-3.5" />
                  <span>上传参考音频</span>
                  <input type="file" accept="audio/*" onChange={(e) => handleFileUpload(e, "tts")} className="hidden" />
                </label>
              </div>

              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">参考音频转录文本 (必须准确匹配参考音频内容)</label>
                <input
                  type="text"
                  value={ttsPromptText}
                  onChange={(e) => setTtsPromptText(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-slate-200 focus:outline-none focus:border-sky-500"
                  placeholder="例如：同时，他强调微调要科学有序。"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">目标合成配音文本</label>
                <textarea
                  rows={3}
                  value={ttsTargetText}
                  onChange={(e) => setTtsTargetText(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200 focus:outline-none focus:border-sky-500"
                  placeholder="输入需要让该音色说出的文本内容..."
                />
              </div>

              <div className="pt-2 border-t border-slate-800 flex items-center space-x-2">
                <input
                  type="text"
                  value={saveVoiceName}
                  onChange={(e) => setSaveVoiceName(e.target.value)}
                  placeholder="将此参考保存为声音模板名称..."
                  className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-2 text-slate-200"
                />
                <button onClick={handleSaveVoice} className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold inline-flex items-center space-x-1">
                  <Save className="w-3.5 h-3.5" />
                  <span>保存声音</span>
                </button>
              </div>
            </div>
          )}

          {/* TAB 3: Speech Edit */}
          {toolSubTab === "edit" && (
            <div className="space-y-4 text-xs">
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setEditType("acoustic")}
                  className={"flex-1 py-2 rounded-xl border text-center font-semibold transition " + (editType === "acoustic" ? "bg-sky-500/10 text-sky-400 border-sky-500/30" : "border-slate-800 text-slate-400 hover:bg-slate-800/50")}
                >
                  声学属性调节 (语速 / 音高 / 音量)
                </button>
                <button
                  onClick={() => setEditType("semantic")}
                  className={"flex-1 py-2 rounded-xl border text-center font-semibold transition " + (editType === "semantic" ? "bg-sky-500/10 text-sky-400 border-sky-500/30" : "border-slate-800 text-slate-400 hover:bg-slate-800/50")}
                >
                  语义文本修改 (删词 / 替换)
                </button>
              </div>

              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">待编辑原音频</label>
                <select
                  value={editAudioId}
                  onChange={(e) => setEditAudioId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  <option value="">-- 选择素材库中的音频 --</option>
                  {assets.map((a) => (
                    <option key={a.id} value={a.id}>{a.name} ({a.duration}s)</option>
                  ))}
                </select>
                <label className="cursor-pointer inline-flex items-center space-x-1 text-sky-400 hover:underline pt-1">
                  <Upload className="w-3.5 h-3.5" />
                  <span>上传待编辑音频</span>
                  <input type="file" accept="audio/*" onChange={(e) => handleFileUpload(e, "edit")} className="hidden" />
                </label>
              </div>

             {editType === "acoustic" ? (
               <div className="space-y-4 pt-2">
                  <div className="flex items-center space-x-2 bg-slate-900 p-1 rounded-xl border border-slate-800">
                    {[
                      { id: "speed", label: "语速调整" },
                      { id: "pitch", label: "音高调整" },
                      { id: "volume", label: "音量微调" },
                      { id: "custom", label: "自定义指令" },
                    ].map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setAcousticMode(m.id as any)}
                        className={"flex-1 py-1 rounded-lg text-center font-medium transition " + (acousticMode === m.id ? "bg-sky-500 text-white font-semibold" : "text-slate-400 hover:text-slate-200")}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>

                  {acousticMode === "speed" && (
                    <div className="space-y-1 p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                      <div className="flex justify-between font-semibold text-slate-300">
                        <span>语速乘数 (Speed: 0.5x ~ 2.0x)</span>
                        <span className="text-sky-400 font-mono">{editSpeed.toFixed(1)}x</span>
                      </div>
                      <input
                        type="range"
                        min={0.5}
                        max={2.0}
                        step={0.1}
                        value={editSpeed}
                        onChange={(e) => setEditSpeed(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg accent-sky-400 cursor-pointer"
                      />
                    </div>
                  )}

                  {acousticMode === "pitch" && (
                    <div className="space-y-1 p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                      <div className="flex justify-between font-semibold text-slate-300">
                        <span>音高半音级数 (Pitch: -6 ~ +6 steps)</span>
                        <span className="text-sky-400 font-mono">{editPitch > 0 ? "+" + editPitch : editPitch}</span>
                      </div>
                      <input
                        type="range"
                        min={-6}
                        max={6}
                        step={1}
                        value={editPitch}
                        onChange={(e) => setEditPitch(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg accent-sky-400 cursor-pointer"
                      />
                    </div>
                  )}

                  {acousticMode === "volume" && (
                    <div className="space-y-1 p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                      <div className="flex justify-between font-semibold text-slate-300">
                        <span>音量增益 (Volume: 0.2x ~ 2.0x)</span>
                        <span className="text-sky-400 font-mono">{editVolume.toFixed(1)}x</span>
                      </div>
                      <input
                        type="range"
                        min={0.2}
                        max={2.0}
                        step={0.1}
                        value={editVolume}
                        onChange={(e) => setEditVolume(parseFloat(e.target.value))}
                        className="w-full h-1.5 bg-slate-800 rounded-lg accent-sky-400 cursor-pointer"
                      />
                    </div>
                  )}

                  {acousticMode === "custom" && (
                    <div className="space-y-1.5">
                      <label className="text-slate-300 font-semibold">自定义声学控制指令</label>
                      <input
                        type="text"
                        value={customAcousticInst}
                        onChange={(e) => setCustomAcousticInst(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-slate-200"
                        placeholder="例如：adjust the speed to 0.7 and shift the pitch by 2 steps"
                      />
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-1.5">
                  <label className="text-slate-300 font-semibold">语义编辑指令</label>
                  <input
                    type="text"
                    value={editInstruction}
                    onChange={(e) => setEditInstruction(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-slate-200 focus:outline-none focus:border-sky-500"
                    placeholder="例如：delete '比普通的茶叶要' 或 substitute 'A' with 'B'"
                  />
                </div>
              )}
            </div>
          )}

          {/* TAB 4: Voice Design */}
          {toolSubTab === "voice_design" && (
            <div className="space-y-4 text-xs">
              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">音色特征自然语言描述</label>
                <textarea
                  rows={3}
                  value={vdInstruction}
                  onChange={(e) => setVdInstruction(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200 focus:outline-none focus:border-sky-500"
                  placeholder="描述您期望的声音特征（性别、年龄、音高、音色质感、情绪、语速等）..."
                />
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {["温柔清晰广播女声", "沉稳磁性青年男声", "活泼开朗元气少女", "严肃庄重新闻播音员"].map((chip) => (
                    <button
                      key={chip}
                      onClick={() => setVdInstruction(chip)}
                      className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-400 text-[10px] border border-slate-800 transition"
                    >
                      + {chip}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-slate-300 font-semibold">目标合成文本</label>
                <textarea
                  rows={3}
                  value={vdText}
                  onChange={(e) => setVdText(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200 focus:outline-none focus:border-sky-500"
                  placeholder="输入让新设计的音色说出的文本内容..."
                />
              </div>
            </div>
          )}

          {/* Advanced Parameters Accordion */}
          <div className="pt-2 border-t border-slate-800">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full flex items-center justify-between text-xs text-slate-400 hover:text-slate-200 py-1"
            >
              <span className="flex items-center space-x-1.5 font-medium">
                <Settings2 className="w-3.5 h-3.5 text-sky-400" />
                <span>高级推理参数 (CFG / Max Steps / Token Limit)</span>
              </span>
              {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {showAdvanced && (
              <div className="mt-3 p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3 text-xs">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <span className="text-slate-400 text-[11px]">文本 Token 上限</span>
                    <input
                      type="number"
                      value={advMaxTextTokens}
                      onChange={(e) => setAdvMaxTextTokens(parseInt(e.target.value) || 300)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-1.5 text-slate-200 font-mono text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <span className="text-slate-400 text-[11px]">最大音频步数</span>
                    <input
                      type="number"
                      value={advMaxAudioSteps}
                      onChange={(e) => setAdvMaxAudioSteps(parseInt(e.target.value) || 750)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-1.5 text-slate-200 font-mono text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <span className="text-slate-400 text-[11px]">CFG 无分类器引导</span>
                    <input
                      type="number"
                      step="0.1"
                      value={advCfg}
                      onChange={(e) => setAdvCfg(parseFloat(e.target.value) || 2.0)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-1.5 text-slate-200 font-mono text-xs"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          <button
            disabled={submitting}
            onClick={handleSubmit}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-bold text-sm shadow-lg shadow-sky-500/25 flex items-center justify-center space-x-2 transition disabled:opacity-50"
          >
            {submitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>任务正在排队处理中...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>立即在 Apple Silicon Metal 上运行推理</span>
              </>
            )}
          </button>
        </div>

        {/* Output Panel */}
        <div className="glass-card rounded-2xl p-6 flex flex-col justify-between space-y-6">
          <div>
            <h3 className="font-bold text-slate-200 text-sm mb-4 flex items-center justify-between">
              <span>推理结果与输出</span>
              {jobResult && (
                <div className="flex items-center space-x-2 text-[11px] font-mono">
                  <span className="text-slate-400">耗时: {jobResult.latency_seconds || 0}s</span>
                  {jobResult.result?.rtf !== undefined && (
                    <span className="px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 font-bold border border-sky-500/30">
                      RTF: {jobResult.result.rtf}
                    </span>
                  )}
                </div>
              )}
            </h3>

            {!jobResult && !submitting && (
              <div className="py-24 text-center text-slate-500 text-xs space-y-2">
                <Sparkles className="w-8 h-8 mx-auto text-slate-600" />
                <p>配置左侧参数后点击运行，生成的语音与文本将在此处实时呈现</p>
              </div>
            )}

            {submitting && (
              <div className="py-24 text-center space-y-3">
                <Loader2 className="w-8 h-8 mx-auto text-sky-400 animate-spin" />
                <p className="text-slate-300 font-semibold text-xs">MLX 神经网络计算中...</p>
                <p className="text-slate-500 text-[11px]">任务 ID: {activeJobId}</p>
              </div>
            )}

            {jobResult && (
              <div className="space-y-4 text-xs">
                {/* Performance Metrics Bar */}
                <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-slate-900/90 border border-slate-800 text-[11px]">
                  <div className="text-center">
                    <span className="text-slate-500 block text-[10px]">推理延迟</span>
                    <span className="font-mono text-slate-200 font-semibold">{jobResult.latency_seconds || 0}s</span>
                  </div>
                  <div className="text-center border-x border-slate-800">
                    <span className="text-slate-500 block text-[10px]">音频时长</span>
                    <span className="font-mono text-slate-200 font-semibold">{jobResult.result?.duration_s || jobResult.result?.audio_duration_s || "--"}s</span>
                  </div>
                  <div className="text-center">
                    <span className="text-slate-500 block text-[10px]">实时率因子 (RTF)</span>
                    <span className="font-mono text-sky-400 font-bold">
                      {jobResult.result?.rtf !== undefined ? `${jobResult.result.rtf} (${(1 / jobResult.result.rtf).toFixed(1)}x)` : "--"}
                    </span>
                  </div>
                </div>

                {/* Audio Output */}
                {jobResult.result?.media_url && (
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-200">生成音频 ({jobResult.result.duration_s}s)</span>
                      <button
                        onClick={() => playAudio(jobResult.result.media_url, "Result-" + jobResult.id, jobResult.result.duration_s)}
                        className="px-3 py-1 rounded-lg bg-sky-500/20 text-sky-300 hover:bg-sky-500/30 transition flex items-center space-x-1"
                      >
                        <Play className="w-3 h-3" />
                        <span>立即播放</span>
                      </button>
                    </div>
                    <audio controls src={jobResult.result.media_url} className="w-full h-8" />
                  </div>
                )}

                {/* Timbre Breakdown for Voice Design */}
                {jobResult.result?.timbre_description && (
                  <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                    <p className="font-semibold text-slate-400 text-[11px]">提取的结构化音色标签</p>
                    <p className="text-slate-200">{jobResult.result.timbre_description}</p>
                  </div>
                )}

                {/* Thinking CoT Accordion */}
                {jobResult.result?.reasoning && (
                  <div className="rounded-xl border border-slate-800 overflow-hidden bg-slate-900/40">
                    <button
                      onClick={() => setShowThinking(!showThinking)}
                      className="w-full px-4 py-2.5 bg-slate-900/80 flex items-center justify-between text-slate-300 font-semibold"
                    >
                      <span>🧠 Chain-of-Thought 思考推导过程</span>
                      {showThinking ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                    </button>
                    {showThinking && (
                      <div className="p-4 text-slate-400 whitespace-pre-wrap leading-relaxed border-t border-slate-800/80 bg-slate-950/40 font-mono text-[11px]">
                        {jobResult.result.reasoning}
                      </div>
                    )}
                  </div>
                )}

                {/* Text Result (ASR / QA Answer / Semantic Text) */}
                {(jobResult.result?.transcript || jobResult.result?.answer || jobResult.result?.rewritten_text) && (
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-slate-400 text-[11px]">识别/回答文本</span>
                      <button
                        onClick={() => handleCopy(jobResult.result.transcript || jobResult.result.answer || jobResult.result.rewritten_text)}
                        className="text-sky-400 hover:underline inline-flex items-center space-x-1"
                      >
                        {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        <span>{copied ? "已复制" : "复制文本"}</span>
                      </button>
                    </div>
                    <p className="text-slate-100 text-sm leading-relaxed whitespace-pre-wrap font-medium">
                      {jobResult.result.transcript || jobResult.result.answer || jobResult.result.rewritten_text}
                    </p>
                  </div>
                )}

                {/* Error Box */}
                {jobResult.error && (
                  <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 space-y-1">
                    <p className="font-bold">任务执行遇到错误</p>
                    <p className="text-[11px]">{jobResult.error}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
