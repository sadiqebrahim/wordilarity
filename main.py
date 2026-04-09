"""
Noun Embedding System
=====================
Compact, fast semantic embeddings for ~10k English nouns.

Pipeline:
  1. Filter GloVe-300d to noun-only vocabulary via WordNet
  2. Compress 300d → 128d with PCA
  3. Save as float16 .npz (~5MB)
  4. Query via pre-normalised cosine similarity (dot product)

Requirements:
    pip install numpy scipy scikit-learn nltk requests tqdm

On first run, downloads:
  - GloVe 840B 300d vectors (~2.2GB, one-time)
  - WordNet via NLTK (small)
"""

import os
import re
import struct
import zipfile
import logging
import urllib.request
from pathlib import Path
from typing import Optional
import sys

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_CACHE = Path("glove.840B.300d.txt")          # raw vectors (large)
MODEL_PATH  = Path("noun_embeddings_google.npz")           # compressed model
DIM_IN      = 300                                   # GloVe native dimension
DIM_OUT     = 128                                   # PCA target dimension
MAX_NOUNS   = 10_000                                # vocabulary cap
MIN_FREQ_RANK = 200_000                             # skip ultra-rare tokens


# ---------------------------------------------------------------------------
# Step 1 — Vocabulary: noun-only wordlist via WordNet
# ---------------------------------------------------------------------------

def get_noun_set() -> set[str]:
    """Return a set of lowercase English nouns from WordNet."""
    import nltk
    from nltk.corpus import wordnet as wn

    for res in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            log.info("Downloading NLTK resource: %s", res)
            nltk.download(res, quiet=True)

    nouns: set[str] = set()
    for synset in wn.all_synsets(pos=wn.NOUN):
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            # Keep only single-token, alphabetic, reasonably short words
            if re.fullmatch(r"[a-z]{3,20}", word):
                nouns.add(word)

    log.info("WordNet noun set: %d unique tokens", len(nouns))
    return nouns


# ---------------------------------------------------------------------------
# Step 2 — GloVe loading with optional download
# ---------------------------------------------------------------------------

def _download_glove(dest: Path) -> None:
    zip_path = dest.with_suffix(".zip")
    if not zip_path.exists():
        log.info("Downloading GloVe (this is ~2.2 GB, one-time) …")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
                  desc="glove.840B.300d.zip") as pbar:
            def reporthook(count, block_size, total_size):
                if pbar.total is None and total_size > 0:
                    pbar.total = total_size
                pbar.update(block_size)
            urllib.request.urlretrieve(GLOVE_URL, zip_path, reporthook)
    log.info("Extracting GloVe …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")


def load_glove_for_nouns(
    noun_set: set[str],
    glove_path: Path = GLOVE_CACHE,
    max_nouns: int = MAX_NOUNS,
) -> tuple[list[str], np.ndarray]:
    """
    Parse GloVe text file; return (words, matrix) for noun_set intersection.
    matrix shape: (N, DIM_IN), dtype float32.
    """
    if not glove_path.exists():
        _download_glove(glove_path)

    words, vecs = [], []
    log.info("Scanning GloVe vectors …")

    with open(glove_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if len(words) >= max_nouns:
                break
            parts = line.rstrip().split(" ")
            token = parts[0].lower()
            if token not in noun_set:
                continue
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if vec.shape[0] != DIM_IN:
                    continue
            except ValueError:
                continue
            words.append(token)
            vecs.append(vec)

            if len(words) % 1000 == 0:
                log.info("  … %d nouns loaded", len(words))

    matrix = np.vstack(vecs)
    log.info("Loaded %d noun vectors, matrix shape %s", len(words), matrix.shape)
    return words, matrix


# ---------------------------------------------------------------------------
# Step 4 — Persist as float16 .npz
# ---------------------------------------------------------------------------


def load_model(path: Path = MODEL_PATH) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    words = data["words"].tolist()
    matrix = data["embeddings"].astype(np.float32)
    log.info("Model loaded: %d nouns × %dd", len(words), matrix.shape[1])
    return words, matrix


# ---------------------------------------------------------------------------
# Step 5 — NounEmbedder: the public API
# ---------------------------------------------------------------------------

class NounEmbedder:
    """
    Fast semantic similarity engine for English nouns.

    Usage
    -----
    emb = NounEmbedder.build()           # first time: downloads GloVe, ~minutes
    emb = NounEmbedder.load()            # subsequent runs: loads .npz, <1s

    emb.most_similar("ocean", n=10)
    emb.similarity("hospital", "clinic")
    emb.score_against("dog", ["cat", "iron", "universe", "puppy"])
    """

    def __init__(self, words: list[str], matrix: np.ndarray) -> None:
        self.words = words
        self._word2idx: dict[str, int] = {w: i for i, w in enumerate(words)}

        # Pre-normalise all vectors → cosine sim becomes a plain dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)          # avoid div-by-zero
        self._unit = (matrix / norms).astype(np.float32)  # shape: (N, D)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------
        

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH) -> "NounEmbedder":
        """Load a previously built model from disk."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run NounEmbedder.build() first."
            )
        words, matrix = load_model(model_path)
        return cls(words, matrix)

    # ------------------------------------------------------------------
    # Core similarity API
    # ------------------------------------------------------------------

    def _vec(self, word: str) -> Optional[np.ndarray]:
        """Return the unit-normed embedding for *word*, or None if OOV."""
        idx = self._word2idx.get(word.lower())
        return self._unit[idx] if idx is not None else None

    def similarity(self, word_a: str, word_b: str) -> float:
        """
        Cosine similarity ∈ [-1, 1] between two nouns.
        Returns NaN if either word is out-of-vocabulary.
        """
        va, vb = self._vec(word_a), self._vec(word_b)
        if va is None or vb is None:
            return float("nan")
        return float(np.dot(va, vb))

    def most_similar(self, word: str, n: int = 50) -> list[tuple[str, float]]:
        """
        Return the *n* nearest nouns to *word* (excluding itself).
        Each element is (noun, cosine_similarity).
        """
        v = self._vec(word)
        if v is None:
            return []
        scores = self._unit @ v                              # (N,) vectorised
        # Exclude the query word itself
        # self_idx = self._word2idx.get(word.lower(), -1)
        # if self_idx >= 0:
        #     scores[self_idx] = -2.0
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.words[i], float(scores[i])) for i in top_idx]
    
    def vec2word(self, vec: Optional[np.ndarray]) -> str:
        # print(self._unit.shape)
        # print(vec.shape)
        scores = self._unit @ vec                              # (N,) vectorised
        top_idx = np.argpartition(scores, -5)[-5:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return self.words[top_idx[0]]
        

    def score_against(
        self, reference: str, candidates: list[str]
    ) -> list[tuple[str, float]]:
        """
        Score a fixed *reference* noun against every word in *candidates*.
        Returns list sorted by descending similarity.
        """
        v = self._vec(reference)
        if v is None:
            return [(c, float("nan")) for c in candidates]

        results = []
        for c in candidates:
            vc = self._vec(c)
            sim = float(np.dot(v, vc)) if vc is not None else float("nan")
            results.append((c, sim))

        results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
        return results

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._word2idx



def random_word_picker(emb: NounEmbedder) -> str:
    vec_dim = emb._unit.shape
    vec = np.random.uniform(-1.5, 1.5, size=(vec_dim[-1],))
    return emb.vec2word(vec)





# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Load a previously built model
    emb = NounEmbedder.load()

    target = random_word_picker(emb)
    

    guesses = 0
    max_guess = 10

    while guesses<max_guess:
        guess = input("Enter Guess: ")
        s = emb.similarity(guess, target)
        bar = "█" * int(max(0, s) * 30)
        print(f"  {guess:>10} × {target:<10}  {s:+.3f}  {bar}")
        print(f"Total guesses: {guesses+1}")
        guesses += 1

        if s==1:
            print("Winner!!!!")
            break
    if guesses == 10:
        print("Lavdena Bhojyam")
        
    print(target)

