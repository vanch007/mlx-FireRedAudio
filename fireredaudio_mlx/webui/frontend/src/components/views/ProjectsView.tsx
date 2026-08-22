import React, { useState } from "react";
import { useApp } from "../../context/AppContext";
import { api } from "../../services/api";
import { FolderPlus, Trash2, Music, Clock, ArrowRight, Play, Upload, Edit3, Check, X, RotateCcw, Mic, MessageSquareQuote, Sliders } from "lucide-react";

export const ProjectsView: React.FC = () => {
  const { projects, activeProjectId, setActiveProjectId, assets, jobs, refreshData, playAudio, sendAssetToTool } = useApp();
  const [name, setName] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [showCreate, setShowCreate] = useState<boolean>(false);
  const [isEditingProj, setIsEditingProj] = useState<boolean>(false);
  const [editName, setEditName] = useState<string>("");
  const [editDesc, setEditDesc] = useState<string>("");

  const handleCreate = async () => {
    if (!name.trim()) return;
    try {
      const p = await api.createProject(name, description);
      setName("");
      setDescription("");
      setShowCreate(false);
      await refreshData();
      setActiveProjectId(p.id);
    } catch (err: any) {
      alert("创建项目失败: " + err.message);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("确定要删除此项目吗？")) return;
    try {
      await api.deleteProject(id);
      if (activeProjectId === id) setActiveProjectId(null);
      await refreshData();
    } catch (err: any) {
      alert("删除失败: " + err.message);
    }
  };

  const handleUpdateProject = async (id: string) => {
    if (!editName.trim()) return;
    try {
      await api.updateProject(id, { name: editName, description: editDesc });
      setIsEditingProj(false);
      await refreshData();
    } catch (err: any) {
      alert("更新失败: " + err.message);
    }
  };

  const handleUploadToProject = async (e: React.ChangeEvent<HTMLInputElement>, projectId: string) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      await api.uploadAsset(file, projectId);
      await refreshData();
    } catch (err: any) {
      alert("上传失败: " + err.message);
    }
  };

  const handleRetryJob = async (jobId: string) => {
    try {
      await api.retryJob(jobId);
      await refreshData();
    } catch (err: any) {
      alert("重试失败: " + err.message);
    }
  };

  const activeProject = projects.find((p) => p.id === activeProjectId);
  const projectAssets = assets.filter((a) => a.project_id === activeProjectId || activeProject?.asset_ids?.includes(a.id));
  const projectJobs = jobs.filter((j) => j.project_id === activeProjectId || activeProject?.job_ids?.includes(j.id));

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">项目工作区管理</h2>
          <p className="text-xs text-slate-400 mt-1">创建长期音频创作项目，集中组织参考音频、合成配音、改写音频与任务记录</p>
        </div>
        <button
          onClick={() => setShowCreate(!showCreate)}
          className="px-4 py-2 rounded-xl bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold shadow-md flex items-center space-x-2 transition"
        >
          <FolderPlus className="w-4 h-4" />
          <span>新建项目</span>
        </button>
      </div>

      {/* Create Modal / Card */}
      {showCreate && (
        <div className="glass-card rounded-2xl p-6 space-y-4 border-sky-500/30">
          <h3 className="font-bold text-slate-200 text-sm">创建新音频项目</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="项目名称 (例如：科技发布会配音、访谈会议纪要)..."
              className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200"
            />
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="项目简要说明 (可选)..."
              className="bg-slate-900 border border-slate-800 rounded-xl p-3 text-slate-200"
            />
          </div>
          <div className="flex justify-end space-x-3 text-xs font-semibold">
            <button onClick={() => setShowCreate(false)} className="px-4 py-2 rounded-xl bg-slate-800 text-slate-300">取消</button>
            <button onClick={handleCreate} className="px-4 py-2 rounded-xl bg-sky-500 text-white">立即创建</button>
          </div>
        </div>
      )}

      {/* Projects List & Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Projects sidebar list */}
        <div className="space-y-3">
          <p className="font-semibold text-slate-400 text-xs uppercase tracking-wider">全部项目 ({projects.length})</p>
          {projects.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-2xl">
              暂无项目，点击右上角新建
            </div>
          ) : (
            projects.map((p) => {
              const isSelected = p.id === activeProjectId;
              return (
                <div
                  key={p.id}
                  onClick={() => setActiveProjectId(p.id)}
                  className={"p-4 rounded-2xl cursor-pointer transition border " + (isSelected ? "bg-sky-500/10 border-sky-500/40 text-white" : "bg-slate-900/50 border-slate-800 text-slate-300 hover:border-slate-700")}
                >
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-bold text-sm truncate">{p.name}</h4>
                    <button onClick={(e) => handleDelete(p.id, e)} className="text-slate-500 hover:text-rose-400 p-1">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <p className="text-[11px] text-slate-400 truncate mb-2">{p.description || "无项目备注"}</p>
                  <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                    <span>{p.asset_ids?.length || 0} 个素材</span>
                    <span>{p.job_ids?.length || 0} 个任务</span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Active Project Workspace */}
        <div className="lg:col-span-2 glass-card rounded-2xl p-6 space-y-6">
          {activeProject ? (
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                {isEditingProj ? (
                  <div className="flex-1 mr-4 space-y-2">
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-1.5 text-sm text-white font-bold"
                    />
                    <input
                      type="text"
                      value={editDesc}
                      onChange={(e) => setEditDesc(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-1.5 text-xs text-slate-300"
                    />
                    <div className="flex items-center space-x-2">
                      <button onClick={() => handleUpdateProject(activeProject.id)} className="px-2.5 py-1 rounded bg-sky-500 text-white text-xs flex items-center space-x-1">
                        <Check className="w-3 h-3" />
                        <span>保存修改</span>
                      </button>
                      <button onClick={() => setIsEditingProj(false)} className="px-2.5 py-1 rounded bg-slate-800 text-slate-400 text-xs">取消</button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-bold text-lg text-white">{activeProject.name}</h3>
                      <button
                        onClick={() => { setEditName(activeProject.name); setEditDesc(activeProject.description); setIsEditingProj(true); }}
                        className="text-slate-500 hover:text-sky-400 p-1"
                        title="编辑项目名称与描述"
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <p className="text-xs text-slate-400">{activeProject.description || "工作区素材与任务已加载"}</p>
                  </div>
                )}
                <div className="flex items-center space-x-2 text-xs">
                  <label className="cursor-pointer px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition flex items-center space-x-1">
                    <Upload className="w-3.5 h-3.5" />
                    <span>添加音频素材</span>
                    <input type="file" accept="audio/*" onChange={(e) => handleUploadToProject(e, activeProject.id)} className="hidden" />
                  </label>
                </div>
              </div>

              {/* Assets in Project */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-bold text-slate-200 text-xs uppercase tracking-wider">关联音频素材 ({projectAssets.length})</h4>
                </div>
                {projectAssets.length === 0 ? (
                  <div className="p-6 text-center text-slate-500 text-xs bg-slate-900/40 rounded-xl">
                    当前项目暂无素材，在工具箱运行生成或上传时会自动关联
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-3">
                    {projectAssets.map((a) => (
                      <div key={a.id} className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between text-xs">
                        <div className="flex items-center space-x-3 truncate">
                          <Music className="w-4 h-4 text-sky-400 shrink-0" />
                          <div>
                            <span className="truncate text-slate-200 font-semibold">{a.name}</span>
                            <p className="text-[10px] text-slate-500">{a.duration}s · {a.sample_rate}Hz</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1.5">
                          <button
                            onClick={() => playAudio(a.media_url, a.name, a.duration)}
                            className="p-1.5 rounded-lg bg-slate-800 hover:bg-sky-500/20 text-sky-300"
                            title="播放"
                          >
                            <Play className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => sendAssetToTool(a.id, 'asr')}
                            className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center space-x-1"
                            title="发送到转写与问答"
                          >
                            <Mic className="w-3 h-3 text-sky-400" />
                            <span>转写</span>
                          </button>
                          <button
                            onClick={() => sendAssetToTool(a.id, 'tts')}
                            className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center space-x-1"
                            title="作为参考音色配音"
                          >
                            <MessageSquareQuote className="w-3 h-3 text-indigo-400" />
                            <span>克隆</span>
                          </button>
                          <button
                            onClick={() => sendAssetToTool(a.id, 'edit_acoustic')}
                            className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center space-x-1"
                            title="编辑音频"
                          >
                            <Sliders className="w-3 h-3 text-amber-400" />
                            <span>编辑</span>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Jobs in Project */}
              <div className="space-y-3 pt-4 border-t border-slate-800">
                <h4 className="font-bold text-slate-200 text-xs uppercase tracking-wider">项目任务执行历史 ({projectJobs.length})</h4>
                {projectJobs.length === 0 ? (
                  <div className="p-6 text-center text-slate-500 text-xs bg-slate-900/40 rounded-xl">暂无执行任务</div>
                ) : (
                  <div className="space-y-2">
                    {projectJobs.map((j) => (
                      <div key={j.id} className="p-3 rounded-xl bg-slate-900/40 border border-slate-800 text-xs flex items-center justify-between">
                        <div>
                          <div className="flex items-center space-x-2">
                            <span className="font-bold text-slate-200 uppercase">{j.task}</span>
                            <span className={"text-[10px] px-1.5 py-0.5 rounded " + (j.status === "completed" ? "bg-emerald-500/20 text-emerald-300" : j.status === "failed" ? "bg-rose-500/20 text-rose-300" : j.status === "interrupted" ? "bg-slate-700 text-slate-300" : "bg-amber-500/20 text-amber-300")}>{j.status}</span>
                          </div>
                          <p className="text-[11px] text-slate-400">{j.current_step}</p>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="font-mono text-slate-500 text-[11px]">{j.latency_seconds ? (j.latency_seconds + "s" + (j.result?.rtf ? " · RTF " + j.result.rtf : "")) : ""}</span>
                          {j.result?.media_url && (
                            <button onClick={() => playAudio(j.result.media_url, "Project-" + j.id, j.result.duration_s)} className="p-1 rounded bg-sky-500/20 text-sky-300">
                              <Play className="w-3.5 h-3.5" />
                            </button>
                          )}
                          {(j.status === "failed" || j.status === "interrupted" || j.status === "cancelled") && (
                            <button onClick={() => handleRetryJob(j.id)} className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300" title="重试任务">
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="py-24 text-center text-slate-500 text-xs">请在左侧选择或新建一个项目以查看工作台详情</div>
          )}
        </div>
      </div>
    </div>
  );
};
