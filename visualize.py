import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import TransformerClassifier


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def tokenize(text: str):
    return text.strip().lower().split()


def encode_text(text: str, vocab: dict[str, int], max_len: int):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab.get(UNK_TOKEN, 1)) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [vocab.get(PAD_TOKEN, 0)] * (max_len - length)
    return ids, tokens[:max_len]


def load_vocab(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_label(label: str):
    return label.strip().capitalize()


def pick_sample_from_test(processed_dir: Path, label_names: list[str]):
    test_data = torch.load(processed_dir / "test.pt")
    label_index = int(test_data["labels"][0].item())
    return test_data["texts"][0], label_names[label_index]


def infer_model_dims(model_name: str):
    if "d128_ff256" in model_name:
        return 128, 256
    if "d32_ff64" in model_name:
        return 32, 64
    return 64, 128


def load_transformer_model(model_path: str | Path, processed_dir: str | Path = "data/processed"):
    processed_dir = Path(processed_dir)
    model_path = Path(model_path)
    vocab = load_vocab(processed_dir / "vocab.json")
    meta = load_meta(processed_dir / "meta.json")
    d_model, d_ff = infer_model_dims(model_path.stem.replace("model_", ""))

    model = TransformerClassifier(
        vocab_size=meta["vocab_size"],
        d_model=d_model,
        d_ff=d_ff,
        max_len=meta["max_len"],
        num_classes=meta["num_classes"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, vocab, meta


def predict_with_attention(sentence: str, model_path: str | Path, processed_dir: str | Path = "data/processed"):
    model, vocab, meta = load_transformer_model(model_path, processed_dir)
    input_ids, tokens = encode_text(sentence, vocab, meta["max_len"])

    with torch.no_grad():
        logits = model(torch.tensor([input_ids], dtype=torch.long))
        pred = logits.argmax(dim=-1).item()
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        weights = model.last_attention_weights[0, : len(tokens), : len(tokens)].cpu().numpy()

    return {
        "sentence": sentence,
        "tokens": tokens,
        "predicted_index": pred,
        "predicted_label": meta["label_names"][pred],
        "probabilities": dict(zip(meta["label_names"], probabilities)),
        "weights": weights,
        "meta": meta,
    }


def create_attention_figure(tokens, weights, predicted_label: str, true_label: str | None = None):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights)
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(len(tokens)), tokens)
    title_lines = []
    if true_label:
        title_lines.append(f"True label: {format_label(true_label)}")
    title_lines.append(f"Predicted label: {format_label(predicted_label)}")
    plt.title("\n".join(title_lines))
    plt.tight_layout()
    return plt.gcf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--sentence", type=str, default="")
    parser.add_argument("--true_label", type=str, default="")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)

    if args.model:
        model_path = Path(args.model)
    else:
        candidates = sorted(results_dir.glob("model_Transformer*.pt"))
        if not candidates:
            raise FileNotFoundError("Khong tim thay model_Transformer*.pt trong results/. Hay chay train.py truoc.")
        model_path = candidates[0]

    meta = load_meta(processed_dir / "meta.json")
    if args.sentence:
        sentence = args.sentence
        true_label = args.true_label.strip() or None
    else:
        sentence, true_label = pick_sample_from_test(processed_dir, meta["label_names"])

    prediction = predict_with_attention(sentence, model_path, processed_dir)

    out_path = results_dir / "attention_heatmap.png"
    fig = create_attention_figure(
        prediction["tokens"],
        prediction["weights"],
        prediction["predicted_label"],
        true_label,
    )
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Sentence: {sentence}")
    if true_label:
        print(f"True label: {format_label(true_label)}")
    print(f"Predicted label: {format_label(prediction['predicted_label'])}")
    print(f"Saved heatmap to: {out_path}")


if __name__ == "__main__":
    main()
