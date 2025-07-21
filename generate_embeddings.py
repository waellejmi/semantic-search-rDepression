import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import numpy as np
from accelerate import Accelerator
from modular.utils import  set_up_info_loggger
from modular.engine import  mean_pooling

logger = set_up_info_loggger()

def get_embeddings(model, tokenizer, texts, batch_size=32, device="cuda"):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def generate_embeddings(model_path: str, data_path: str, save_dir: str, with_faiss: bool = True):
    accelerator = Accelerator()
    checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    accelerator.load_state(model_path)
    model = accelerator.prepare(model)
    model.eval()
    device = accelerator.device

    logger.info(f"Loading dataset from {data_path}")
    dataset = Dataset.from_csv(data_path)
    dataset = dataset.remove_columns(["positive","negative"])

    logger.info(f"Loaded {len(dataset)} entries.")

    logger.info("Computing embeddings...")
    embeddings = get_embeddings(model, tokenizer, dataset["text"], batch_size=32, device=device)

    dataset = dataset.add_column("embeddings", embeddings.tolist())

    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)
    logger.info(f"Dataset with embeddings saved to {save_dir}")

    if with_faiss:
        logger.info("Building FAISS index...")
        dataset.add_faiss_index(column="embeddings")
        dataset.save_faiss_index("embeddings", os.path.join(save_dir, "faiss.index"))
        logger.info("FAISS index saved.")

    logger.info("Embedding generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save embeddings + FAISS index.")
    parser.add_argument("--model_path", type=str, default="models/best_model_checkpoint", help="Path to trained model")
    parser.add_argument("--data_path", type=str, default="data/r_depression_posts.csv", help="CSV dataset path")
    parser.add_argument("--save_dir", type=str, default="data/with_embeddings", help="Where to save embeddings and FAISS")
    parser.add_argument("--no_faiss", action="store_true", help="Disable FAISS index creation")
    args = parser.parse_args()

    generate_embeddings(
        model_path=args.model_path,
        data_path=args.data_path,
        save_dir=args.save_dir,
        with_faiss=not args.no_faiss
    )
