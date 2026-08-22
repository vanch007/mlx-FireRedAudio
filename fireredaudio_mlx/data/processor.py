"""Audio processor using WhisperFeatureExtractor."""

from typing import List, Union, Dict, Any
import numpy as np
from transformers import WhisperFeatureExtractor

from ..utils.audio import UNDERSTAND_SAMPLE_RATE


class FireRedAudioProcessor:
    def __init__(self, feature_extractor: WhisperFeatureExtractor):
        self.feature_extractor = feature_extractor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        fe = WhisperFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        return cls(fe)

    def __call__(
        self,
        audios: Union[np.ndarray, List[np.ndarray]],
        max_length: int = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if isinstance(audios, np.ndarray) and audios.ndim == 1:
            audios = [audios]

        kwargs["padding"] = "max_length"
        kwargs["sampling_rate"] = UNDERSTAND_SAMPLE_RATE
        kwargs["return_attention_mask"] = True
        kwargs["return_tensors"] = "np"
        if max_length is not None:
            kwargs["max_length"] = max_length

        res = self.feature_extractor(audios, **kwargs)
        res["feature_attention_mask"] = res.pop("attention_mask")
        return res
