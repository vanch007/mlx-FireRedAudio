import React, { useRef, useState, useEffect } from "react";
import { useApp } from "../../context/AppContext";
import { Play, Pause, X, Volume2, Download, Gauge } from "lucide-react";

export const AudioPlayer: React.FC = () => {
  const { playback, stopAudio, toggleAudio } = useApp();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [volume, setVolume] = useState<number>(1);
  const [playbackRate, setPlaybackRate] = useState<number>(1);

  useEffect(() => {
    if (audioRef.current && playback) {
      if (playback.isPlaying) {
        audioRef.current.play().catch(() => {});
      } else {
        audioRef.current.pause();
      }
    }
  }, [playback]);

  if (!playback) return null;

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
      setDuration(audioRef.current.duration || 0);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = val;
      setCurrentTime(val);
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    setVolume(v);
    if (audioRef.current) {
      audioRef.current.volume = v;
    }
  };

  const cycleSpeed = () => {
    const speeds = [0.8, 1.0, 1.25, 1.5, 2.0];
    const next = speeds[(speeds.indexOf(playbackRate) + 1) % speeds.length];
    setPlaybackRate(next);
    if (audioRef.current) {
      audioRef.current.playbackRate = next;
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs)) return "0:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  };

  return (
    <div className="fixed bottom-4 left-72 right-8 z-50 glass-card rounded-2xl p-4 shadow-2xl flex items-center space-x-6 border border-slate-700/80">
      <audio
        ref={audioRef}
        src={playback.url}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleTimeUpdate}
        onEnded={() => toggleAudio()}
        autoPlay
      />
      <button
        onClick={toggleAudio}
        className="w-12 h-12 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 text-white flex items-center justify-center shadow-lg shadow-sky-500/30 hover:scale-105 active:scale-95 transition"
      >
        {playback.isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
      </button>
      <div className="flex-1 space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-200 truncate max-w-md">{playback.name}</span>
          <span className="text-slate-400 font-mono text-[11px]">
            {formatTime(currentTime)} / {formatTime(duration || playback.duration || 0)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={duration || playback.duration || 100}
          step={0.01}
          value={currentTime}
          onChange={handleSeek}
          className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-sky-400"
        />
      </div>
      <div className="flex items-center space-x-3 text-slate-400">
        <button
          onClick={cycleSpeed}
          className="flex items-center space-x-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-sky-400 transition"
          title="切换播放倍速"
        >
          <Gauge className="w-3.5 h-3.5" />
          <span>{playbackRate}x</span>
        </button>
        <div className="flex items-center space-x-1.5">
          <Volume2 className="w-4 h-4 text-slate-400" />
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            onChange={handleVolumeChange}
            className="w-16 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-sky-400"
          />
        </div>
        <a
          href={playback.url}
          download={playback.name}
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
          title="下载音频文件"
        >
          <Download className="w-4 h-4" />
        </a>
        <button
          onClick={stopAudio}
          className="p-2 rounded-lg bg-slate-800 hover:bg-rose-500/20 hover:text-rose-400 text-slate-400 transition"
          title="关闭播放器"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
