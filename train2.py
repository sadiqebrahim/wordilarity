import gensim
import re
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

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

# Path to Google News Word2Vec model
model_path = "GoogleNews-vectors-negative300.bin.gz"

print("Loading Word2Vec model...")
model = gensim.models.KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

print("Model loaded!")

# Your noun set (example)
# Replace this with your actual noun list
noun_set = get_noun_set()

# Storage
rows = []

output_file = "noun_embeddings_google.txt"

missing_words = []

with open(output_file, "w", encoding="utf-8") as f:
    for word in tqdm(noun_set):
        token = word.replace(" ", "_")  # match Word2Vec format

        if token in model:
            vector = model[token]
            vector_str = " ".join(map(str, vector))
            f.write(f"{token} {vector_str}\n")
        else:
            missing_words.append(word)

print(f"Saved embeddings to {output_file}")
print(f"Missing words: {len(missing_words)}")

