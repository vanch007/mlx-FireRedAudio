"""FireRedAudio MLX: Standalone Apple Silicon Audio Language Model Package."""

from .inference import FireRedAudioInference, UnderstandOutput, AudioOutput
from .loader import load_mlx_fireredaudio
from .models.modeling import FireRedAudioModel

__version__ = "0.1.0"
__all__ = [
    "FireRedAudioInference",
    "FireRedAudioModel",
    "UnderstandOutput",
    "AudioOutput",
    "load_mlx_fireredaudio",
]
