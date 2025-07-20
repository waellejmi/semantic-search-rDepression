import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import json
from modular.data_setup import TripletDataset, triplet_collate_fn, get_train_test_val_dataset,build_ranking_test_dataset
from modular.engine import process_triplet_batch, mean_pooling
from modular.utils import *

from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

logger = set_up_info_loggger()


def evaluate_ranking(model, tokenizer, examples, top_k=5):
    all_mrr = []
    all_recall_at_k = []

    model.eval()
    with torch.no_grad():
        for ex in tqdm(examples, desc="Ranking Eval"):
            texts = [ex['anchor']] + ex['candidates']
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)

            outputs = model(**encoded)
            embeddings = F.normalize(
                mean_pooling(outputs, encoded['attention_mask']), p=2, dim=1
            )

            anchor_emb = embeddings[0].unsqueeze(0)
            candidate_embs = embeddings[1:]
            sims = F.cosine_similarity(anchor_emb, candidate_embs).cpu().numpy()

            rank = (-sims).argsort()
            rank_of_true = np.where(rank == ex['label'])[0][0] + 1
            all_mrr.append(1.0 / rank_of_true)
            all_recall_at_k.append(1.0 if rank_of_true <= top_k else 0.0)

    return {
        'MRR': np.mean(all_mrr),
        f'Recall@{top_k}': np.mean(all_recall_at_k)
    }


def run_inference_with_dataloader(model_path, test_dataset, batch_size=8):
    accelerator = Accelerator()

    checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    accelerator.load_state(model_path)

    test_triplet_dataset = TripletDataset(test_dataset, tokenizer)
    test_loader = DataLoader(
        test_triplet_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=triplet_collate_fn
    )

    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    all_predictions, all_pos_sim, all_neg_sim, all_margins = [], [], [], []

    logger.info("Running triplet-based evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            anchor_emb, pos_emb, neg_emb = process_triplet_batch(model, batch)
            pos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=1)
            neg_sim = F.cosine_similarity(anchor_emb, neg_emb, dim=1)
            predictions = (pos_sim > neg_sim).float()
            margins = pos_sim - neg_sim

            all_predictions.extend(predictions.cpu().numpy())
            all_pos_sim.extend(pos_sim.cpu().numpy())
            all_neg_sim.extend(neg_sim.cpu().numpy())
            all_margins.extend(margins.cpu().numpy())

    results = {
        'predictions': np.array(all_predictions),
        'pos_similarities': np.array(all_pos_sim),
        'neg_similarities': np.array(all_neg_sim),
        'margins': np.array(all_margins),
        'accuracy': np.mean(all_predictions),
        'mean_positive_similarity': np.mean(all_pos_sim),
        'mean_negative_similarity': np.mean(all_neg_sim),
        'mean_margin': np.mean(all_margins),
        'std_margin': np.std(all_margins),
        'model': model,
        'tokenizer': tokenizer
    }
    return results


def main():
    DATA_PATH = "data/r_depression_posts.csv"
    MODEL_PATH = "models/best_model_checkpoint"
    BATCH_SIZE = 8
    RESULTS_PATH = "results/test_results.json"
    PLOTS_PATH = "results/test_results_plots.png"

    dataset = get_train_test_val_dataset(DATA_PATH)
    test_dataset = dataset['test']

    logger.info("Starting inference...")
    results = run_inference_with_dataloader(MODEL_PATH, test_dataset, BATCH_SIZE)

    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"Mean Positive Similarity: {results['mean_positive_similarity']:.4f}")
    logger.info(f"Mean Negative Similarity: {results['mean_negative_similarity']:.4f}")
    logger.info(f"Mean Margin: {results['mean_margin']:.4f}")
    logger.info(f"Std Margin: {results['std_margin']:.4f}")

    correct = int(np.sum(results['predictions']))
    total = len(results['predictions'])
    wrong = total - correct
    logger.info(f"\nCorrect: {correct}/{total}, Incorrect: {wrong}")

    low_margin = np.sum(results['margins'] < 0.1)
    logger.info(f"Challenging cases (margin < 0.1): {low_margin}/{total} ({low_margin/total*100:.2f}%)")
    logger.info("="*50)

    # Ranking evaluation
    ranking_data = build_ranking_test_dataset(test_dataset, num_negatives=5)
    ranking_metrics = evaluate_ranking(results['model'], results['tokenizer'], ranking_data)
    logger.info(f"MRR: {ranking_metrics['MRR']:.4f}")
    logger.info(f"Recall@5: {ranking_metrics['Recall@5']:.4f}")

    # Save results
    results_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.floating) else v)
                       for k, v in results.items() if k != 'model' and k != 'tokenizer'}
    results_to_save.update(ranking_metrics)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"Results saved to {RESULTS_PATH}")

    visualize_results(results, logger, PLOTS_PATH)
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
