"""Thread-safe single-instance ModelManager for FireRedAudio."""

import asyncio
import logging
import os
import threading
import time
from typing import Optional, Dict, Any
import mlx.core as mx

from ..inference import FireRedAudioInference
from .config import DEFAULT_MODEL_DIR
from .sse import broadcaster

logger = logging.getLogger(__name__)


def init_mlx_thread():
    """Initialize CPU and GPU streams in the current thread to prevent stream errors on CPU fallbacks."""
    try:
        mx.default_stream(mx.cpu)
        mx.default_stream(mx.gpu)
        mx.set_default_device(mx.gpu)
    except Exception:
        pass

class ModelManager:
    def __init__(self, default_model_path: Optional[str] = None):
        self.model_path = str(default_model_path or DEFAULT_MODEL_DIR)
        self.engine: Optional[FireRedAudioInference] = None
        self.status: str = "unloaded"  # unloaded | loading | ready | error
        self.load_error: Optional[str] = None
        self.load_time_seconds: float = 0.0
        self._lock = threading.Lock()
        self._loading_thread: Optional[threading.Thread] = None

    def get_memory_info(self) -> Dict[str, Any]:
        """Fetch Apple Silicon Metal and process memory metrics."""
        info: Dict[str, Any] = {
            "device": "Apple Silicon Metal",
            "mlx_version": mx.__version__,
            "active_memory_gb": 0.0,
            "peak_memory_gb": 0.0,
            "cache_memory_gb": 0.0,
        }
        try:
            dev_info = mx.device_info()
            info["device_name"] = dev_info.get("device_name", "Apple GPU")
            info["max_recommended_gb"] = round(dev_info.get("max_recommended_working_set_size", 0) / (1024**3), 2)
            info["total_memory_gb"] = round(dev_info.get("memory_size", 0) / (1024**3), 2)
        except Exception:
            pass

        try:
            active = mx.get_active_memory() if hasattr(mx, "get_active_memory") else mx.metal.get_active_memory()
            peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
            cache = mx.get_cache_memory() if hasattr(mx, "get_cache_memory") else mx.metal.get_cache_memory()
            info["active_memory_gb"] = round(active / (1024**3), 2)
            info["peak_memory_gb"] = round(peak / (1024**3), 2)
            info["cache_memory_gb"] = round(cache / (1024**3), 2)
        except Exception:
            pass

        return info

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_path": self.model_path,
            "load_time_seconds": round(self.load_time_seconds, 4),
            "error": self.load_error,
            "memory": self.get_memory_info(),
        }

    def _load_worker(self):
        with self._lock:
            if self.status == "ready" and self.engine is not None:
                return
            self.status = "loading"
            self.load_error = None
            logger.info("Background loading FireRedAudio model from %s...", self.model_path)
            t0 = time.time()
            try:
                init_mlx_thread()
                self.engine = FireRedAudioInference(model_path=self.model_path)
                self.load_time_seconds = time.time() - t0
                self.status = "ready"
                logger.info("FireRedAudio model loaded successfully in %.4fs", self.load_time_seconds)
            except Exception as e:
                self.status = "error"
                self.load_error = str(e)
                logger.error("Failed to load FireRedAudio model: %s", e, exc_info=True)

    def start_background_load(self):
        """Trigger model load asynchronously on application startup."""
        if self.status in ("loading", "ready"):
            return
        self.status = "loading"
        self._loading_thread = threading.Thread(target=self._load_worker, daemon=True)
        self._loading_thread.start()

    def reload_model(self, new_model_path: str):
        """Unload current model and reload from a new model path (e.g. 8-bit or 4-bit)."""
        with self._lock:
            self.engine = None
            self.model_path = new_model_path
            self.status = "unloaded"
            self.load_error = None
            self.clear_cache()
        self.start_background_load()

    def ensure_ready(self, timeout: float = 300.0) -> FireRedAudioInference:
        """Ensure the engine is loaded, waiting if necessary."""
        if self.status == "ready" and self.engine is not None:
            return self.engine

        if self.status == "unloaded":
            self.start_background_load()

        start_wait = time.time()
        while self.status == "loading":
            if time.time() - start_wait > timeout:
                raise TimeoutError("Timed out waiting for FireRedAudio model to load.")
            time.sleep(0.5)

        if self.status == "error":
            raise RuntimeError(f"FireRedAudio model failed to load: {self.load_error}")

        if self.engine is None:
            raise RuntimeError("Engine instance is None despite ready status.")
        return self.engine

    def clear_cache(self):
        try:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
        except Exception:
            pass


model_manager = ModelManager()
