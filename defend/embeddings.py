"""Cached CPU MiniLM embeddings shared by semantic features and precedent memory."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class CachedMiniLM:
    def __init__(self, cache_path: str | Path, model_name: str = MODEL_NAME) -> None:
        self.cache_path = Path(cache_path)
        self.model_name = model_name
        self._vectors: dict[str, np.ndarray] = {}
        self._model = None
        self._dirty = False
        if self.cache_path.is_file():
            payload = np.load(self.cache_path, allow_pickle=False)
            self._vectors = {
                str(key): vector.astype(np.float32, copy=False)
                for key, vector in zip(payload["keys"], payload["vectors"], strict=True)
            }

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _encoder(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device="cpu", local_files_only=True
            )
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = [str(text or "").strip() for text in texts]
        if not values:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        missing: dict[str, str] = {}
        for value in values:
            key = self._key(value)
            if key not in self._vectors:
                missing[key] = value
        if missing:
            vectors = self._encoder().encode(
                list(missing.values()),
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32, copy=False)
            self._vectors.update(zip(missing, vectors, strict=True))
            self._dirty = True
        return np.vstack([self._vectors[self._key(value)] for value in values]).astype(np.float32, copy=False)

    def flush(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self._vectors)
        vectors = np.vstack([self._vectors[key] for key in keys]).astype(np.float32, copy=False)
        temporary = self.cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, keys=np.asarray(keys), vectors=vectors)
        os.replace(temporary, self.cache_path)
        self._dirty = False

    def __enter__(self) -> "CachedMiniLM":
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()
