from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sentencepiece as spm
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizers"
DATA_DIR = OUTPUT_DIR / "data"

# I kept the language set small on purpose so the comparison stays readable
# while still covering different scripts and morphology profiles.
LANGUAGES = {
    "en": "English",
    "tr": "Turkish",
    "sw": "Swahili",
    "am": "Amharic",
}

RANDOM_STATE = 42
# This mimics a low-resource setup even though the benchmark itself is larger.
LOW_RESOURCE_TRAIN_SIZE = 3000
TOKENIZER_VOCAB_SIZE = 800
WORD_VOCAB_SIZE = 800


@dataclass
class WordLevelTokenizer:
    vocab: set[str]
    unk_token: str = "<unk>"

    def encode(self, text: str) -> list[str]:
        # Anything outside the fixed vocabulary is treated as unknown, which is
        # exactly the weakness we want to compare against subword models.
        return [token if token in self.vocab else self.unk_token for token in text.split()]


@dataclass
class SentencePieceTokenizer:
    processor: spm.SentencePieceProcessor

    def encode(self, text: str) -> list[str]:
        return self.processor.encode(text, out_type=str)


def ensure_directories() -> None:
    for directory in (OUTPUT_DIR, FIGURE_DIR, TOKENIZER_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def sample_low_resource_split(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    if size >= len(frame):
        return frame.copy()

    # Stratification keeps the label mix stable after downsampling.
    sampled, _ = train_test_split(
        frame,
        train_size=size,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )
    return sampled.reset_index(drop=True)


def load_language_dataframe(language_code: str) -> dict[str, pd.DataFrame]:
    dataset = load_dataset("mteb/MassiveIntentClassification", language_code)
    frames = {
        split: pd.DataFrame(dataset[split])[["text", "label"]]
        for split in ("train", "validation", "test")
    }
    return frames


def build_word_tokenizer(texts: Iterable[str], vocab_size: int) -> WordLevelTokenizer:
    counts = Counter()
    for text in texts:
        counts.update(text.split())

    # The word-level baseline is deliberately simple: keep only the most
    # frequent tokens and let everything else fall back to <unk>.
    most_common = [token for token, _ in counts.most_common(vocab_size - 1)]
    return WordLevelTokenizer(vocab=set(most_common))


def build_sentencepiece_tokenizer(
    texts: Iterable[str],
    language_code: str,
    model_type: str,
    vocab_size: int,
) -> SentencePieceTokenizer:
    model_prefix = TOKENIZER_DIR / f"{language_code}_{model_type}"
    # SentencePiece expects a plain-text corpus, so we write one temporary file
    # per language/tokenizer pair and delete it right after training.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
        for text in texts:
            handle.write(text.replace("\n", " ").strip() + "\n")
        corpus_path = Path(handle.name)

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=0,
        unk_id=1,
        train_extremely_large_corpus=False,
    )
    corpus_path.unlink(missing_ok=True)

    processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
    return SentencePieceTokenizer(processor=processor)


def word_can_be_encoded(tokenizer, word: str) -> bool:
    pieces = tokenizer.encode(word)
    return "<unk>" not in pieces


def compute_intrinsic_metrics(language: str, tokenizer_name: str, tokenizer, texts: Iterable[str]) -> dict:
    total_tokens = 0
    total_unk = 0
    fertility_values = []
    chars_per_token_values = []
    word_coverages = []

    for text in texts:
        whitespace_words = text.split()
        pieces = tokenizer.encode(text)
        if not pieces:
            continue

        total_tokens += len(pieces)
        total_unk += sum(piece == "<unk>" for piece in pieces)
        # Fertility tells us how aggressively a tokenizer breaks words apart.
        fertility_values.append(len(pieces) / max(1, len(whitespace_words)))
        chars_per_token_values.append(len(text) / len(pieces))

        if whitespace_words:
            # This is a word-level view of coverage: can each whitespace word be
            # represented without triggering an unknown token?
            word_coverages.append(
                sum(word_can_be_encoded(tokenizer, word) for word in whitespace_words) / len(whitespace_words)
            )

    return {
        "language": language,
        "tokenizer": tokenizer_name,
        "oov_piece_rate": total_unk / max(1, total_tokens),
        "avg_fertility": sum(fertility_values) / max(1, len(fertility_values)),
        "avg_chars_per_token": sum(chars_per_token_values) / max(1, len(chars_per_token_values)),
        "avg_word_coverage": sum(word_coverages) / max(1, len(word_coverages)),
    }


def evaluate_downstream(tokenizer_name: str, tokenizer, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    # We convert each sentence into tokenizer-specific "surface text" so the
    # same TF-IDF + SVM pipeline can be reused across all tokenizer variants.
    train_texts = [" ".join(tokenizer.encode(text)) for text in train_df["text"]]
    test_texts = [" ".join(tokenizer.encode(text)) for text in test_df["text"]]

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    classifier = LinearSVC(random_state=RANDOM_STATE)
    classifier.fit(x_train, train_df["label"])
    predictions = classifier.predict(x_test)

    return {
        "tokenizer": tokenizer_name,
        "macro_f1": f1_score(test_df["label"], predictions, average="macro"),
        "accuracy": accuracy_score(test_df["label"], predictions),
    }


def save_figures(intrinsic_df: pd.DataFrame, downstream_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    coverage_fig = plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=intrinsic_df, x="language", y="avg_word_coverage", hue="tokenizer")
    ax.set_title("Word Coverage on Held-Out Test Data")
    ax.set_ylabel("Coverage")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    coverage_fig.tight_layout()
    coverage_fig.savefig(FIGURE_DIR / "coverage_by_language.png", dpi=200)
    plt.close(coverage_fig)

    fertility_fig = plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=intrinsic_df, x="language", y="avg_fertility", hue="tokenizer")
    ax.set_title("Token Fertility by Language")
    ax.set_ylabel("Subword pieces per whitespace word")
    ax.set_xlabel("")
    fertility_fig.tight_layout()
    fertility_fig.savefig(FIGURE_DIR / "fertility_by_language.png", dpi=200)
    plt.close(fertility_fig)

    f1_fig = plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=downstream_df, x="language", y="macro_f1", hue="tokenizer")
    ax.set_title("Downstream Intent Classification Macro-F1")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    f1_fig.tight_layout()
    f1_fig.savefig(FIGURE_DIR / "macro_f1_by_language.png", dpi=200)
    plt.close(f1_fig)


def build_dataset_profile(language_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_lengths = train_df["text"].str.split().map(len)
    test_lengths = test_df["text"].str.split().map(len)
    # Unique training words are useful as a rough proxy for lexical variety.
    unique_words = len({word for text in train_df["text"] for word in text.split()})
    return {
        "language": language_name,
        "train_examples": len(train_df),
        "test_examples": len(test_df),
        "avg_train_words": train_lengths.mean(),
        "avg_test_words": test_lengths.mean(),
        "unique_train_words": unique_words,
        "labels": train_df["label"].nunique(),
    }


def main() -> None:
    ensure_directories()

    dataset_profiles = []
    intrinsic_rows = []
    downstream_rows = []
    summary_rows = []

    for language_code, language_name in LANGUAGES.items():
        frames = load_language_dataframe(language_code)
        train_df = sample_low_resource_split(frames["train"], LOW_RESOURCE_TRAIN_SIZE)
        test_df = frames["test"].reset_index(drop=True)

        dataset_profiles.append(build_dataset_profile(language_name, train_df, test_df))

        # Every language gets the same tokenizer budget so the comparison stays fair.
        tokenizers = {
            "word": build_word_tokenizer(train_df["text"], WORD_VOCAB_SIZE),
            "bpe": build_sentencepiece_tokenizer(train_df["text"], language_code, "bpe", TOKENIZER_VOCAB_SIZE),
            "unigram": build_sentencepiece_tokenizer(train_df["text"], language_code, "unigram", TOKENIZER_VOCAB_SIZE),
        }

        for tokenizer_name, tokenizer in tokenizers.items():
            intrinsic_result = compute_intrinsic_metrics(language_name, tokenizer_name, tokenizer, test_df["text"])
            downstream_result = evaluate_downstream(tokenizer_name, tokenizer, train_df, test_df)
            downstream_result["language"] = language_name

            intrinsic_rows.append(intrinsic_result)
            downstream_rows.append(downstream_result)

        # Merge intrinsic and downstream scores so we can declare one "best"
        # tokenizer per language in the final summary table.
        merged = (
            pd.DataFrame([row for row in intrinsic_rows if row["language"] == language_name])
            .merge(
                pd.DataFrame([row for row in downstream_rows if row["language"] == language_name]),
                on=["language", "tokenizer"],
            )
            .sort_values("macro_f1", ascending=False)
        )
        summary_rows.append(
            {
                "language": language_name,
                "best_tokenizer": merged.iloc[0]["tokenizer"],
                "best_macro_f1": merged.iloc[0]["macro_f1"],
                "word_macro_f1": merged.loc[merged["tokenizer"] == "word", "macro_f1"].iloc[0],
                "bpe_macro_f1": merged.loc[merged["tokenizer"] == "bpe", "macro_f1"].iloc[0],
                "unigram_macro_f1": merged.loc[merged["tokenizer"] == "unigram", "macro_f1"].iloc[0],
            }
        )

    dataset_df = pd.DataFrame(dataset_profiles)
    intrinsic_df = pd.DataFrame(intrinsic_rows).sort_values(["language", "tokenizer"]).reset_index(drop=True)
    downstream_df = pd.DataFrame(downstream_rows).sort_values(["language", "tokenizer"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("language").reset_index(drop=True)

    dataset_df.to_csv(DATA_DIR / "dataset_profile.csv", index=False)
    intrinsic_df.to_csv(DATA_DIR / "intrinsic_metrics.csv", index=False)
    downstream_df.to_csv(DATA_DIR / "downstream_metrics.csv", index=False)
    summary_df.to_csv(DATA_DIR / "summary_metrics.csv", index=False)

    save_figures(intrinsic_df, downstream_df)

    narrative = {
        "project_title": "Multilingual Tokenization",
        "languages": LANGUAGES,
        "train_size_per_language": LOW_RESOURCE_TRAIN_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "summary": summary_df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "results_summary.json").write_text(json.dumps(narrative, indent=2), encoding="utf-8")

    print("Experiment complete.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
