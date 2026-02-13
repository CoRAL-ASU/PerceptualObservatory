import os
import json
import argparse 
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
from datasets import load_from_disk, Dataset
from tqdm import tqdm


# ---------- Utilities ----------
def b64_to_image(b64_str: str) -> Image.Image:
    """Decode base64 image to PIL."""
    if b64_str and b64_str.strip().lower().startswith("data:image"):
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(BytesIO(data)).convert("RGB")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", required=True, help="Path to HuggingFace dataset (saved with save_to_disk)")
    ap.add_argument("--output", required=True, help="Output JSONL of embeddings")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32", type=str, help="Model name or path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32, help="Number of images per GPU batch")
    ap.add_argument("--text-batch-size", type=int, default=64, help="Number of texts per GPU batch")
    ap.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    ap.add_argument("--save", type=int, default=500, help="Save every N samples")
    args = ap.parse_args()

    # -------------------- Model setup --------------------
    if args.model.startswith("openai/clip"):
        model = CLIPModel.from_pretrained(args.model).to(args.device)
        processor = CLIPProcessor.from_pretrained(args.model)

        def encode_text(texts):
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(args.device)
            with torch.no_grad():
                return model.get_text_features(**inputs).to(torch.float32).cpu().numpy()

        def encode_image(pils):
            inputs = processor(images=pils, return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                return model.get_image_features(**inputs).to(torch.float32).cpu().numpy()

    elif args.model.startswith("google/siglip"):
        model = AutoModel.from_pretrained(args.model, device_map="auto").eval()
        processor = AutoProcessor.from_pretrained(args.model)

        def encode_text(texts):
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                return model.get_text_features(**inputs).to(torch.float32).cpu().numpy()

        def encode_image(pils):
            inputs = processor(images=pils, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                return model.get_image_features(**inputs).to(torch.float32).cpu().numpy()

    elif args.model == "nvidia/MM-Embed":
        from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig

        # Properly remove LlavaNextConfig from AutoModel._model_mapping to avoid duplicate registration
        try:
            if hasattr(AutoModel, "_model_mapping") and hasattr(AutoModel._model_mapping, "_mapping"):
                mapping = AutoModel._model_mapping._mapping
                if LlavaNextConfig in mapping:
                    del mapping[LlavaNextConfig]
                    print("[INFO] Removed duplicate LlavaNextConfig from AutoModel._model_mapping")
        except Exception as e:
            print(f"[WARN] Could not fully patch AutoModel: {e}")

        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device)
        max_length = 4096

        def encode_text(texts):
            passages = [{"txt": t} for t in texts]
            with torch.no_grad():
                out = model.encode(passages, is_query=False, max_length=max_length)
            return out["hidden_states"].cpu().numpy()

        def encode_image(pils):
            passages = [{"img": pil} for pil in pils]
            with torch.no_grad():
                out = model.encode(passages, is_query=False, max_length=max_length)
            return out["hidden_states"].cpu().numpy()



    elif args.model.startswith("vidore/colpali"):
        from colpali_engine.models import ColPali, ColPaliProcessor

        device_map = "auto" if args.device.startswith("cuda") else args.device
        model = ColPali.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()
        processor = ColPaliProcessor.from_pretrained(args.model)

        def encode_text(texts):
            batch_queries = processor.process_queries(texts).to(model.device)
            with torch.no_grad():
                q_emb = model(**batch_queries)
            return q_emb.to(torch.float32).cpu().numpy()

        def encode_image(pils):
            batch_images = processor.process_images(pils).to(model.device)
            with torch.no_grad():
                i_emb = model(**batch_images)
            return i_emb.to(torch.float32).cpu().numpy()

    elif args.model.startswith("jinaai/jina-embeddings-v4"):
        model = AutoModel.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.float32
        ).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        def encode_text(texts):
            return model.encode_text(texts=texts, task="retrieval", prompt_name="query")

        def encode_image(pils):
            return model.encode_image(images=pils, task="retrieval")

    else:
        raise ValueError(f"Model {args.model} not supported.")

    # -------------------- Load dataset --------------------
    dataset: Dataset = load_from_disk(args.hf_dataset)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"[INFO] Loaded dataset with {len(dataset)} samples")

    # -------------------- Process in batches --------------------
    buffer, batch_images, batch_keys, batch_entries = [], [], [], []
    text_batch, text_entries = [], []

    fout = open(args.output, "w")

    for idx, rec in enumerate(tqdm(dataset, desc="Processing")):
        entry = {"id": rec["id"], "text_emb": None, "img_embs": {}}

        # ---- Collect text ----
        if rec.get("label"):
            text_batch.append(rec["label"])
            text_entries.append(entry)

        # ---- Collect images ----
        images_to_encode, image_keys = [], []
        if rec.get("gt"):
            images_to_encode.append(b64_to_image(rec["gt"]))
            image_keys.append(("gt", None))

        for k, aug in (rec.get("augmentation") or {}).items():
            if "img" in aug:
                images_to_encode.append(b64_to_image(aug["img"]))
                image_keys.append(("aug", k))

        for k, ill in (rec.get("illusion") or {}).items():
            if "img" in ill:
                images_to_encode.append(b64_to_image(ill["img"]))
                image_keys.append(("illusion", k))

        if images_to_encode:
            batch_images.extend(images_to_encode)
            batch_keys.extend([(entry, typ, key) for (typ, key) in image_keys])

        batch_entries.append(entry)

        # ---- Process text batch ----
        if len(text_batch) >= args.text_batch_size or idx == len(dataset) - 1:
            if text_batch:
                t_embs = encode_text(text_batch)
                for ent, emb in zip(text_entries, t_embs):
                    ent["text_emb"] = emb.tolist() if hasattr(emb, "tolist") else emb
            text_batch, text_entries = [], []

        # ---- Process image batch ----
        if len(batch_images) >= args.batch_size or idx == len(dataset) - 1:
            if batch_images:
                img_embs = encode_image(batch_images)
                for (ent, typ, key), emb in zip(batch_keys, img_embs):
                    emb = emb.tolist() if hasattr(emb, "tolist") else emb
                    if typ == "gt":
                        ent["img_embs"]["gt"] = emb
                    else:
                        ent["img_embs"][key] = emb
            buffer.extend(batch_entries)
            batch_images, batch_keys, batch_entries = [], [], []

        # ---- Periodic save ----
        if (idx + 1) % args.save == 0 or idx == len(dataset) - 1:
            for e in buffer:
                fout.write(json.dumps(e) + "\n")
            fout.flush()
            buffer = []

    fout.close()
    print(f"[✓] Saved embeddings for {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()
