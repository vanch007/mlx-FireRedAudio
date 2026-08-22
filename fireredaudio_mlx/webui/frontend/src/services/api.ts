import { SystemStatus, QualityPreset, Asset, VoiceProfile, Project, Job } from '../types';

const API_BASE = '/api/v1';

export const api = {
  // System & Hardware
  getSystemStatus: async (): Promise<SystemStatus> => {
    const res = await fetch(`${API_BASE}/system/status`);
    if (!res.ok) throw new Error('Failed to fetch system status');
    return res.json();
  },

  getPresets: async (): Promise<Record<string, QualityPreset>> => {
    const res = await fetch(`${API_BASE}/system/presets`);
    if (!res.ok) throw new Error('Failed to fetch presets');
    return res.json();
  },

  clearCache: async (): Promise<{ message: string }> => {
    const res = await fetch(`${API_BASE}/system/cache/clear`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to clear cache');
    return res.json();
  },

  loadModel: async (modelPath: string): Promise<{ message: string; status: SystemStatus }> => {
    const res = await fetch(`${API_BASE}/system/model/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath }),
    });
    if (!res.ok) throw new Error('Failed to trigger model load');
    return res.json();
  },

  // Assets
  listAssets: async (projectId?: string): Promise<Asset[]> => {
    const url = projectId ? `${API_BASE}/assets?project_id=${projectId}` : `${API_BASE}/assets`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch assets');
    return res.json();
  },

  uploadAsset: async (file: File, projectId?: string, source: string = 'upload'): Promise<Asset> => {
    const formData = new FormData();
    formData.append('file', file);
    if (projectId) formData.append('project_id', projectId);
    formData.append('source', source);

    const res = await fetch(`${API_BASE}/assets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to upload asset');
    return res.json();
  },

  deleteAsset: async (assetId: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/assets/${assetId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete asset');
  },

  // Voices
  listVoices: async (): Promise<VoiceProfile[]> => {
    const res = await fetch(`${API_BASE}/voices`);
    if (!res.ok) throw new Error('Failed to fetch voices');
    return res.json();
  },

  createVoice: async (data: { name: string; prompt_text: string; audio_asset_id: string; language: string; description?: string }): Promise<VoiceProfile> => {
    const res = await fetch(`${API_BASE}/voices`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to create voice');
    return res.json();
  },

  deleteVoice: async (voiceId: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/voices/${voiceId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete voice');
  },

  // Projects
  listProjects: async (): Promise<Project[]> => {
    const res = await fetch(`${API_BASE}/projects`);
    if (!res.ok) throw new Error('Failed to fetch projects');
    return res.json();
  },

  createProject: async (name: string, description: string = ''): Promise<Project> => {
    const res = await fetch(`${API_BASE}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    if (!res.ok) throw new Error('Failed to create project');
    return res.json();
  },

  updateProject: async (projectId: string, data: { name?: string; description?: string }): Promise<Project> => {
    const res = await fetch(`${API_BASE}/projects/${projectId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to update project');
    return res.json();
  },

  deleteProject: async (projectId: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/projects/${projectId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete project');
  },

  // Jobs
  listJobs: async (projectId?: string): Promise<Job[]> => {
    const url = projectId ? `${API_BASE}/jobs?project_id=${projectId}` : `${API_BASE}/jobs`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch jobs');
    return res.json();
  },

  createJob: async (task: string, params: Record<string, any>, projectId?: string, preset: string = 'balanced'): Promise<Job> => {
    const res = await fetch(`${API_BASE}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, params, project_id: projectId, preset }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to submit job' }));
      throw new Error(err.detail || 'Failed to submit job');
    }
    return res.json();
  },

  cancelJob: async (jobId: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to cancel job');
  },

  retryJob: async (jobId: string): Promise<Job> => {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/retry`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Failed to retry job' }));
      throw new Error(err.detail || 'Failed to retry job');
    }
    return res.json();
  },

  getResult: async (resultId: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/results/${resultId}`);
    if (!res.ok) throw new Error('Failed to fetch result');
    return res.json();
  },
};

export function createEventSource(onEvent: (event: string, data: any) => void) {
  const es = new EventSource('/api/v1/events');
  es.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data);
      onEvent('message', parsed);
    } catch {}
  };
  es.addEventListener('system_status', (e: any) => {
    try {
      onEvent('system_status', JSON.parse(e.data));
    } catch {}
  });
  es.addEventListener('job_created', (e: any) => {
    try {
      onEvent('job_created', JSON.parse(e.data));
    } catch {}
  });
  es.addEventListener('job_progress', (e: any) => {
    try {
      onEvent('job_progress', JSON.parse(e.data));
    } catch {}
  });
  es.addEventListener('job_completed', (e: any) => {
    try {
      onEvent('job_completed', JSON.parse(e.data));
    } catch {}
  });
  es.addEventListener('job_failed', (e: any) => {
    try {
      onEvent('job_failed', JSON.parse(e.data));
    } catch {}
  });
  es.addEventListener('job_cancelled', (e: any) => {
    try {
      onEvent('job_cancelled', JSON.parse(e.data));
    } catch {}
  });
  return es;
}
