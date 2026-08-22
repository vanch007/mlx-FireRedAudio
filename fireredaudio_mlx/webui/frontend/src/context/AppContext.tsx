import React, { createContext, useContext, useState, useEffect } from 'react';
import { SystemStatus, QualityPreset, Asset, VoiceProfile, Project, Job } from '../types';
import { api, createEventSource } from '../services/api';

interface AudioPlayback {
  url: string;
  name: string;
  duration?: number;
  isPlaying: boolean;
}

export interface ToolPrefill {
  asrAudioIds?: string[];
  asrTask?: 'asr' | 'understand';
  ttsPromptAudioId?: string;
  ttsPromptText?: string;
  ttsLanguage?: string;
  editAudioId?: string;
  editType?: 'acoustic' | 'semantic';
}

interface AppContextType {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  toolSubTab: string;
  setToolSubTab: (tab: string) => void;
  systemStatus: SystemStatus | null;
  presets: Record<string, QualityPreset>;
  projects: Project[];
  assets: Asset[];
  voices: VoiceProfile[];
  jobs: Job[];
  activeProjectId: string | null;
  setActiveProjectId: (id: string | null) => void;
  playback: AudioPlayback | null;
  playAudio: (url: string, name: string, duration?: number) => void;
  stopAudio: () => void;
  toggleAudio: () => void;
  refreshData: () => Promise<void>;
  clearCache: () => Promise<void>;
  toolPrefill: ToolPrefill;
  setToolPrefill: (prefill: ToolPrefill | ((prev: ToolPrefill) => ToolPrefill)) => void;
  sendAssetToTool: (assetId: string, tool: 'asr' | 'understand' | 'tts' | 'edit_acoustic' | 'edit_semantic') => void;
  sendVoiceToTTS: (voice: VoiceProfile) => void;
}

const AppContext = createContext<AppContextType | null>(null);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentTab, setCurrentTab] = useState<string>('overview');
  const [toolSubTab, setToolSubTab] = useState<string>('asr_qa');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [presets, setPresets] = useState<Record<string, QualityPreset>>({});
  const [projects, setProjects] = useState<Project[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [voices, setVoices] = useState<VoiceProfile[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [playback, setPlayback] = useState<AudioPlayback | null>(null);
  const [toolPrefill, setToolPrefill] = useState<ToolPrefill>({});

  const refreshData = async () => {
    try {
      const [statusRes, presetsRes, projectsRes, assetsRes, voicesRes, jobsRes] = await Promise.all([
        api.getSystemStatus().catch(() => null),
        api.getPresets().catch(() => ({})),
        api.listProjects().catch(() => []),
        api.listAssets().catch(() => []),
        api.listVoices().catch(() => []),
        api.listJobs().catch(() => []),
      ]);
      if (statusRes) setSystemStatus(statusRes);
      setPresets(presetsRes);
      setProjects(projectsRes);
      setAssets(assetsRes);
      setVoices(voicesRes);
      setJobs(jobsRes);
    } catch (e) {
      console.error('Failed to refresh data', e);
    }
  };

  useEffect(() => {
    refreshData();

    // Connect SSE stream
    const es = createEventSource((event, data) => {
      if (event === 'system_status') {
        setSystemStatus(data);
      } else if (event === 'job_created' || event === 'job_progress' || event === 'job_completed' || event === 'job_failed' || event === 'job_cancelled') {
        setJobs((prev) => {
          const idx = prev.findIndex((j) => j.id === data.id);
          if (idx >= 0) {
            const copy = [...prev];
            copy[idx] = data;
            return copy;
          } else {
            return [data, ...prev];
          }
        });
        if (event === 'job_completed') {
          // refresh assets list if new audio was created
          api.listAssets().then(setAssets).catch(() => {});
        }
      }
    });

    // Periodic memory poll every 5s
    const timer = setInterval(() => {
      api.getSystemStatus().then(setSystemStatus).catch(() => {});
    }, 5000);

    return () => {
      es.close();
      clearInterval(timer);
    };
  }, []);

  const playAudio = (url: string, name: string, duration?: number) => {
    setPlayback({ url, name, duration, isPlaying: true });
  };

  const stopAudio = () => {
    setPlayback(null);
  };

  const toggleAudio = () => {
    if (playback) {
      setPlayback({ ...playback, isPlaying: !playback.isPlaying });
    }
  };

  const clearCache = async () => {
    await api.clearCache();
    const s = await api.getSystemStatus();
    setSystemStatus(s);
  };

  const sendAssetToTool = (assetId: string, tool: 'asr' | 'understand' | 'tts' | 'edit_acoustic' | 'edit_semantic') => {
    if (tool === 'asr' || tool === 'understand') {
      setToolPrefill((prev) => ({ ...prev, asrAudioIds: [assetId], asrTask: tool }));
      setToolSubTab('asr_qa');
    } else if (tool === 'tts') {
      setToolPrefill((prev) => ({ ...prev, ttsPromptAudioId: assetId }));
      setToolSubTab('tts');
    } else if (tool === 'edit_acoustic' || tool === 'edit_semantic') {
      setToolPrefill((prev) => ({
        ...prev,
        editAudioId: assetId,
        editType: tool === 'edit_acoustic' ? 'acoustic' : 'semantic',
      }));
      setToolSubTab('edit');
    }
    setCurrentTab('tools');
  };

  const sendVoiceToTTS = (voice: VoiceProfile) => {
    setToolPrefill((prev) => ({
      ...prev,
      ttsPromptAudioId: voice.audio_asset_id,
      ttsPromptText: voice.prompt_text,
      ttsLanguage: voice.language,
    }));
    setToolSubTab('tts');
    setCurrentTab('tools');
  };

  return (
    <AppContext.Provider
      value={{
        currentTab,
        setCurrentTab,
        toolSubTab,
        setToolSubTab,
        systemStatus,
        presets,
        projects,
        assets,
        voices,
        jobs,
        activeProjectId,
        setActiveProjectId,
        playback,
        playAudio,
        stopAudio,
        toggleAudio,
        refreshData,
        clearCache,
        toolPrefill,
        setToolPrefill,
        sendAssetToTool,
        sendVoiceToTTS,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
};
