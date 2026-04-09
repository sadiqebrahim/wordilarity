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

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_CACHE = Path("google_300d.txt")          # raw vectors (large)
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
# Step 3 — PCA compression  (300d → 128d)
# ---------------------------------------------------------------------------

def compress_pca(
    matrix: np.ndarray,
    n_components: int = DIM_OUT,
) -> tuple[np.ndarray, "sklearn.decomposition.PCA"]:
    """
    Fit PCA on matrix, return (compressed_matrix, pca_model).
    compressed_matrix shape: (N, n_components), dtype float32.
    """
    from sklearn.decomposition import PCA

    log.info("Fitting PCA %dd → %dd …", DIM_IN, n_components)
    pca = PCA(n_components=n_components, random_state=42)
    compressed = pca.fit_transform(matrix).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    log.info("PCA explained variance: %.1f%%", explained * 100)
    return compressed, pca


# ---------------------------------------------------------------------------
# Step 4 — Persist as float16 .npz
# ---------------------------------------------------------------------------

def save_model(
    words: list[str],
    matrix: np.ndarray,
    path: Path = MODEL_PATH,
) -> None:
    """Save vocabulary + embedding matrix as a compact float16 .npz."""
    mat16 = matrix.astype(np.float16)
    np.savez_compressed(
        path,
        words=np.array(words, dtype=object),
        embeddings=mat16,
    )
    size_mb = path.stat().st_size / 1e6
    log.info("Model saved: %s  (%.1f MB, %d nouns × %dd)", path, size_mb, len(words), matrix.shape[1])


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
    def build(
        cls,
        model_path: Path = MODEL_PATH,
        dim_out: int = DIM_OUT,
        max_nouns: int = MAX_NOUNS,
    ) -> "NounEmbedder":
        """Full pipeline: GloVe → noun filter → PCA → save → return instance."""
        noun_set = get_noun_set()
        words, matrix = load_glove_for_nouns(noun_set, max_nouns=len(noun_set))
        compressed, _ = compress_pca(matrix, n_components=dim_out)
        save_model(words, compressed, path=model_path)
        return cls(words, compressed)
        # save_model(words, matrix, path=model_path)
        # return cls(words, matrix)
        

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

    def most_similar(self, word: str, n: int = 10) -> list[tuple[str, float]]:
        """
        Return the *n* nearest nouns to *word* (excluding itself).
        Each element is (noun, cosine_similarity).
        """
        v = self._vec(word)
        if v is None:
            return []
        print(self._unit.shape)
        print(v.shape)
        scores = self._unit @ v                              # (N,) vectorised
        # Exclude the query word itself
        # self_idx = self._word2idx.get(word.lower(), -1)
        # if self_idx >= 0:
        #     scores[self_idx] = -2.0
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.words[i], float(scores[i])) for i in top_idx]

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


# ---------------------------------------------------------------------------
# Step 6 — Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_wordsim353(emb: NounEmbedder, path: str = "wordsim353.csv") -> float:
    """
    Compute Spearman correlation against WordSim-353.
    Download from: https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip

    Returns Spearman r, or NaN if the file is unavailable.
    """
    from scipy.stats import spearmanr

    if not os.path.exists(path):
        log.warning("WordSim-353 file not found at %s — skipping evaluation.", path)
        return float("nan")

    human, model = [], []
    with open(path) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            w1, w2, score = parts[0].lower(), parts[1].lower(), float(parts[2])
            sim = emb.similarity(w1, w2)
            if not np.isnan(sim):
                human.append(score)
                model.append(sim)

    r, p = spearmanr(human, model)
    log.info("WordSim-353 Spearman r = %.3f  (n=%d pairs)", r, len(human))
    return r


def audit_nearest_neighbors(emb: NounEmbedder, probes: list[str], n: int = 8) -> None:
    """Print nearest-neighbor spot-check for a list of probe words."""
    print("\n── Nearest-neighbor audit ──────────────────────────────────")
    for word in probes:
        if word not in emb:
            print(f"  {word!r:>14}  [OOV]")
            continue
        neighbors = emb.most_similar(word, n=n)
        nbr_str = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
        print(f"  {word!r:>14}  →  {nbr_str}")
    print()


def cluster_audit(emb: NounEmbedder, k: int = 12, sample: int = 200) -> None:
    """Run k-means on a sample of embeddings and print cluster summaries."""
    from sklearn.cluster import KMeans

    log.info("Running k-means (k=%d) on %d sampled nouns …", k, sample)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb.words), size=min(sample, len(emb.words)), replace=False)
    sub_words = [emb.words[i] for i in idx]
    sub_vecs  = emb._unit[idx]

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(sub_vecs)

    print("\n── Cluster audit (sample) ───────────────────────────────────")
    for c in range(k):
        members = [sub_words[i] for i, l in enumerate(labels) if l == c]
        print(f"  Cluster {c:>2}:  {', '.join(members[:8])}")
    print()


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def _demo(emb: NounEmbedder) -> None:
    """Interactive demo showcasing the embedding API."""

    print(f"\n{'═'*58}")
    print(f"  NounEmbedder — {len(emb):,} nouns × {emb._unit.shape[1]}d")
    print(f"{'═'*58}\n")

    # 1. Pairwise similarities
    pairs = [
        ("dog",       "cat"),
        ("dog",       "iron"),
        ("hospital",  "clinic"),
        ("hospital",  "mountain"),
        ("ocean",     "sea"),
        ("ocean",     "keyboard"),
        ("king",      "queen"),
        ("man",       "woman"),
        ("king",      "man"),
        ("queen",       "woman")
    ]
    print("── Pairwise cosine similarity ───────────────────────────────")
    for a, b in pairs:
        s = emb.similarity(a, b)
        bar = "█" * int(max(0, s) * 30)
        print(f"  {a:>10} × {b:<10}  {s:+.3f}  {bar}")

    # 2. Nearest neighbors
    audit_nearest_neighbors(emb, ["ocean", "hospital", "king", "computer", "tree"])

    # 3. Ranked scoring against a reference
    reference = "vehicle"
    candidates = ["car", "bicycle", "boat", "horse", "road", "engine",
                  "forest", "happiness", "truck", "wheel"]
    print(f"── Score candidates against reference: {reference!r} ─────────────")
    for word, score in emb.score_against(reference, candidates):
        bar = "█" * int(max(0, score) * 40)
        print(f"  {word:<12}  {score:+.3f}  {bar}")
    print()

    # 4. Clustering audit
    cluster_audit(emb, k=8, sample=160)


# ---------------------------------------------------------------------------
# Lightweight demo using a small synthetic fallback
# (runs without downloading GloVe — uses random vectors for illustration only)
# ---------------------------------------------------------------------------

def _synthetic_demo() -> None:
    """
    Builds a tiny toy model from curated word pairs so the code runs
    without the GloVe download. Semantic quality is illustrative only.
    """
    log.info("Building synthetic demo model (no GloVe required) …")

    # Hand-crafted semantic groups — each group gets a base vector + noise
    groups = {
        "animals":    ["dog", "cat", "horse", "wolf", "fox", "deer", "lion",
                       "tiger", "bear", "rabbit", "fish", "bird"],
        "vehicles":   ["car", "truck", "bicycle", "boat", "ship", "train",
                       "plane", "bus", "motorcycle", "tractor"],
        "buildings":  ["house", "hospital", "school", "church", "library",
                       "factory", "prison", "palace", "temple", "barn"],
        "nature":     ["ocean", "mountain", "forest", "river", "desert",
                       "island", "volcano", "glacier", "lake", "canyon"],
        "food":       ["bread", "meat", "fruit", "rice", "soup", "cheese",
                       "butter", "sugar", "salt", "milk"],
        "technology": ["computer", "phone", "keyboard", "screen", "server",
                       "network", "robot", "camera", "battery", "chip"],
    }

    rng = np.random.default_rng(42)
    D = 64
    words, vecs = [], []

    for group_vec in [rng.standard_normal(D) for _ in groups]:
        for w in list(groups.values())[0 if not words else -1]:
            pass  # placeholder — rebuilt below

    words = []
    vecs  = []
    for base_vec, members in zip(
        [rng.standard_normal(D) * 3 for _ in groups], groups.values()
    ):
        for w in members:
            words.append(w)
            vecs.append(base_vec + rng.standard_normal(D) * 0.5)

    matrix = np.array(vecs, dtype=np.float32)
    emb = NounEmbedder(words, matrix)

    print("\n[Synthetic demo — random vectors per semantic group]")
    _demo(emb)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if mode == "build":
        # Full pipeline: download GloVe, build + save model
        log.info("Building full model from GloVe …")
        emb = NounEmbedder.build()
        _demo(emb)
        evaluate_wordsim353(emb)

    elif mode == "load":
        # Load a previously built model
        emb = NounEmbedder.load()
        _demo(emb)
        evaluate_wordsim353(emb)

    elif mode == "demo":
        # Synthetic demo (no download required)
        _synthetic_demo()

    else:
        # Auto: use saved model if present, else synthetic demo
        if MODEL_PATH.exists():
            log.info("Found saved model — loading.")
            emb = NounEmbedder.load()
            _demo(emb)
        else:
            log.info("No saved model found. Running synthetic demo.")
            log.info("To build the real model, run:  python noun_embeddings.py build")
            _synthetic_demo()