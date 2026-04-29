import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def tokenize(text: str):
    return text.strip().lower().split()


def build_vocab(df: pd.DataFrame):
    counter = Counter()
    for text in df["text"]:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(counter.keys()):
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - length)
    return ids, length


def dataframe_to_tensor_dict(df: pd.DataFrame, vocab: dict[str, int], max_len: int):
    input_ids, lengths, labels, texts = [], [], [], []
    for _, row in df.iterrows():
        ids, length = encode_text(row["text"], vocab, max_len)
        input_ids.append(ids)
        lengths.append(length)
        labels.append(int(row["label"]))
        texts.append(row["text"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
    }


def validate_dataframe(df: pd.DataFrame):
    required_cols = {"id", "split", "text", "label", "label_name", "num_tokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def split_dataframe(df: pd.DataFrame):
    return {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }


def summarize_splits(splits: dict[str, pd.DataFrame]):
    stats = {}
    for name, split_df in splits.items():
        counts = split_df["label_name"].value_counts().to_dict()
        stats[name] = {
            "num_samples": len(split_df),
            "label_counts": {
                "negative": counts.get("negative", 0),
                "neutral": counts.get("neutral", 0),
                "positive": counts.get("positive", 0),
            },
        }
    return stats


def prepare_datasets(data_csv: str | Path = "data/sentiment_raw.csv", output_dir: str | Path = "data/processed", max_len: int = 20):
    data_csv = Path(data_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    validate_dataframe(df)

    splits = split_dataframe(df)
    vocab = build_vocab(splits["train"])

    train_data = dataframe_to_tensor_dict(splits["train"], vocab, max_len)
    val_data = dataframe_to_tensor_dict(splits["val"], vocab, max_len)
    test_data = dataframe_to_tensor_dict(splits["test"], vocab, max_len)

    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data, output_dir / "val.pt")
    torch.save(test_data, output_dir / "test.pt")

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    meta = {
        "max_len": max_len,
        "vocab_size": len(vocab),
        "pad_id": vocab[PAD_TOKEN],
        "unk_id": vocab[UNK_TOKEN],
        "num_classes": 3,
        "label_names": ["negative", "neutral", "positive"],
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "dataframe": df,
        "splits": splits,
        "stats": summarize_splits(splits),
        "vocab": vocab,
        "meta": meta,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data/sentiment_raw.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--show_stats", action="store_true")
    args = parser.parse_args()

    result = prepare_datasets(args.data_csv, args.output_dir, args.max_len)
    splits = result["splits"]
    stats = result["stats"]
    vocab = result["vocab"]

    if args.show_stats:
        for name in ["train", "val", "test"]:
            split_stats = stats[name]
            counts = split_stats["label_counts"]
            print(
                f"[{name.upper()}] {split_stats['num_samples']} mau | "
                f"negative: {counts['negative']} "
                f"neutral: {counts['neutral']} "
                f"positive: {counts['positive']}"
            )
        print(f"Vocab size: {len(vocab)} tu")
        print(f"Tao ra: {result['output_dir'] / 'train.pt'}, val.pt, test.pt, vocab.json, meta.json")


if __name__ == "__main__":
    main()
