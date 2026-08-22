export interface MemoryInfo {
  device: string;
  device_name?: string;
  mlx_version?: string;
  active_memory_gb: number;
  peak_memory_gb: number;
  cache_memory_gb: number;
  max_recommended_gb?: number;
  total_memory_gb?: number;
}

export interface SystemStatus {
  status: 'unloaded' | 'loading' | 'ready' | 'error';
  model_path: string;
  load_time_seconds: number;
  error?: string | null;
  memory: MemoryInfo;
}

export interface QualityPreset {
  id: string;
  name: string;
  description: string;
  n_timesteps: number;
  asr_beam_size: number;
  temperature: number;
  top_k: number;
  top_p: number;
  inference_cfg: number;
}

export interface Asset {
  id: string;
  name: string;
  file_path: string;
  media_url: string;
  file_size: number;
  duration: number;
  sample_rate: number;
  channels: number;
  source: 'upload' | 'generation' | 'reference';
  project_id?: string | null;
  created_at: number;
}

export interface VoiceProfile {
  id: string;
  name: string;
  prompt_text: string;
  audio_asset_id: string;
  language: string;
  description?: string;
  created_at: number;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  asset_ids: string[];
  job_ids: string[];
  created_at: number;
  updated_at: number;
}

export interface Job {
  id: string;
  task: 'asr' | 'understand' | 'tts' | 'edit_acoustic' | 'edit_semantic' | 'voice_design';
  status: 'queued' | 'loading' | 'preprocessing' | 'inferencing' | 'exporting' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  progress: number;
  current_step: string;
  preset?: string;
  params: Record<string, any>;
  result?: Record<string, any> | null;
  error?: string | null;
  latency_seconds: number;
  project_id?: string | null;
  created_at: number;
  started_at?: number | null;
  completed_at?: number | null;
}
