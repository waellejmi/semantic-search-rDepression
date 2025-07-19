#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import DatasetDict
from modular.data_setup import TripletDataset, triplet_collate_fn
from modular.engine import mean_pooling

from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path, tokenizer_path=None):
        """
        Initialize the inference class
        
        Args:
            model_path: Path to the saved model checkpoint
            tokenizer_path: Path to tokenizer (if None, uses same as model_path)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def get_embeddings(self, texts, batch_size=32):
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = mean_pooling(outputs, encoded['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def calculate_similarities(self, anchor_texts, positive_texts, negative_texts):
        """
        Calculate similarities for triplets
        
        Args:
            anchor_texts: List of anchor texts
            positive_texts: List of positive texts  
            negative_texts: List of negative texts
            
        Returns:
            Dictionary with similarity scores and predictions
        """
        logger.info("Calculating embeddings for triplets...")
        
        # Get embeddings
        anchor_embeddings = self.get_embeddings(anchor_texts)
        positive_embeddings = self.get_embeddings(positive_texts)
        negative_embeddings = self.get_embeddings(negative_texts)
        
        # Calculate similarities
        pos_similarities = np.sum(anchor_embeddings * positive_embeddings, axis=1)
        neg_similarities = np.sum(anchor_embeddings * negative_embeddings, axis=1)
        
        # Predictions (1 if positive similarity > negative similarity, 0 otherwise)
        predictions = (pos_similarities > neg_similarities).astype(int)
        
        results = {
            'anchor_texts': anchor_texts,
            'positive_texts': positive_texts,
            'negative_texts': negative_texts,
            'pos_similarities': pos_similarities,
            'neg_similarities': neg_similarities,
            'predictions': predictions,
            'margin': pos_similarities - neg_similarities
        }
        
        return results
    
    def evaluate_triplet_accuracy(self, results):
        """
        Evaluate triplet accuracy
        
        Args:
            results: Dictionary from calculate_similarities
            
        Returns:
            Dictionary with metrics
        """
        predictions = results['predictions']
        accuracy = np.mean(predictions)
        
        pos_similarities = results['pos_similarities']
        neg_similarities = results['neg_similarities']
        margins = results['margin']
        
        metrics = {
            'accuracy': accuracy,
            'mean_positive_similarity': np.mean(pos_similarities),
            'mean_negative_similarity': np.mean(neg_similarities),
            'mean_margin': np.mean(margins),
            'std_margin': np.std(margins),
            'min_margin': np.min(margins),
            'max_margin': np.max(margins)
        }
        
        return metrics


def load_test_data(data_path):
    """Load and prepare test data"""
    logger.info(f"Loading test data from {data_path}")
    
    # Load dataset
    ds = DatasetDict.from_csv(data_path)
    ds = ds.remove_columns("Unnamed: 0")
    
    # Split data (same as training)
    train_val, test = ds.train_test_split(test_size=0.03, seed=42).values()
    
    logger.info(f"Test dataset size: {len(test)}")
    return test


def run_inference_with_dataloader(model_path, test_dataset, batch_size=8):
    """
    Run inference using the same dataloader setup as training
    
    Args:
        model_path: Path to saved model
        test_dataset: Test dataset
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with results
    """
    # Load model using Accelerator (same as training)
    accelerator = Accelerator()
    
    # Load tokenizer and model
    checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    
    # Load trained weights
    accelerator.load_state(model_path)
    
    # Prepare test data
    test_triplet_dataset = TripletDataset(test_dataset, tokenizer)
    test_loader = DataLoader(
        test_triplet_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=triplet_collate_fn
    )
    
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()
    
    # Run inference
    all_predictions = []
    all_pos_similarities = []
    all_neg_similarities = []
    all_margins = []
    
    logger.info("Running inference on test dataset...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Process batch efficiently (same as training)
            all_input_ids = torch.cat([
                batch['anchor_input_ids'],
                batch['positive_input_ids'], 
                batch['negative_input_ids']
            ], dim=0)
            
            all_attention_mask = torch.cat([
                batch['anchor_attention_mask'],
                batch['positive_attention_mask'],
                batch['negative_attention_mask']
            ], dim=0)
            
            # Forward pass
            outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
            embeddings = mean_pooling(outputs, all_attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Split embeddings
            batch_size_actual = batch['anchor_input_ids'].size(0)
            anchor_emb = embeddings[:batch_size_actual]
            positive_emb = embeddings[batch_size_actual:2*batch_size_actual]
            negative_emb = embeddings[2*batch_size_actual:]
            
            # Calculate similarities
            pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
            neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1)
            
            # Store results
            predictions = (pos_sim > neg_sim).float()
            margins = pos_sim - neg_sim
            
            all_predictions.extend(predictions.cpu().numpy())
            all_pos_similarities.extend(pos_sim.cpu().numpy())
            all_neg_similarities.extend(neg_sim.cpu().numpy())
            all_margins.extend(margins.cpu().numpy())
    
    # Calculate metrics
    results = {
        'predictions': np.array(all_predictions),
        'pos_similarities': np.array(all_pos_similarities),
        'neg_similarities': np.array(all_neg_similarities),
        'margins': np.array(all_margins),
        'accuracy': np.mean(all_predictions),
        'mean_positive_similarity': np.mean(all_pos_similarities),
        'mean_negative_similarity': np.mean(all_neg_similarities),
        'mean_margin': np.mean(all_margins),
        'std_margin': np.std(all_margins)
    }
    
    return results


def visualize_results(results, save_path=None):
    """
    Create visualizations of the results
    
    Args:
        results: Results dictionary
        save_path: Path to save plots (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Similarity distributions
    axes[0, 0].hist(results['pos_similarities'], alpha=0.7, label='Positive', bins=50, color='green')
    axes[0, 0].hist(results['neg_similarities'], alpha=0.7, label='Negative', bins=50, color='red')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Similarities')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Margin distribution
    axes[0, 1].hist(results['margins'], bins=50, alpha=0.7, color='blue')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
    axes[0, 1].set_xlabel('Margin (Pos - Neg Similarity)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Margins')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot of similarities
    axes[1, 0].scatter(results['neg_similarities'], results['pos_similarities'], 
                      alpha=0.6, s=20)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect separation')
    axes[1, 0].set_xlabel('Negative Similarity')
    axes[1, 0].set_ylabel('Positive Similarity')
    axes[1, 0].set_title('Positive vs Negative Similarities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics
    accuracy = results['accuracy']
    mean_margin = results['mean_margin']
    std_margin = results['std_margin']
    
    metrics_text = f"""
    Accuracy: {accuracy:.4f}
    Mean Margin: {mean_margin:.4f}
    Std Margin: {std_margin:.4f}
    Mean Pos Sim: {results['mean_positive_similarity']:.4f}
    Mean Neg Sim: {results['mean_negative_similarity']:.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                   verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to {save_path}")
    
    plt.show()


def main():
    """Main function to run the inference"""
    
    # Configuration
    DATA_PATH = "data/r_depression_posts.csv"
    MODEL_PATH = "best_model_checkpoint"  # or "final_model_checkpoint"
    BATCH_SIZE = 8
    RESULTS_PATH = "test_results.json"
    PLOTS_PATH = "test_results_plots.png"
    
    # Load test data
    test_dataset = load_test_data(DATA_PATH)
    
    # Run inference
    logger.info("Starting inference...")
    results = run_inference_with_dataloader(MODEL_PATH, test_dataset, BATCH_SIZE)
    
    # Print results with additional analysis
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS")
    logger.info("="*50)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"Mean Positive Similarity: {results['mean_positive_similarity']:.4f}")
    logger.info(f"Mean Negative Similarity: {results['mean_negative_similarity']:.4f}")
    logger.info(f"Mean Margin: {results['mean_margin']:.4f}")
    logger.info(f"Std Margin: {results['std_margin']:.4f}")
    
    # Additional analysis
    correct_predictions = np.sum(results['predictions'])
    total_predictions = len(results['predictions'])
    incorrect_predictions = total_predictions - correct_predictions
    
    logger.info(f"\nDetailed Analysis:")
    logger.info(f"Correct predictions: {int(correct_predictions)}/{total_predictions}")
    logger.info(f"Incorrect predictions: {int(incorrect_predictions)}")
    logger.info(f"Separation quality: {results['mean_margin']:.4f} ± {results['std_margin']:.4f}")
    
    # Check for challenging cases
    challenging_cases = np.sum(results['margins'] < 0.1)
    logger.info(f"Challenging cases (margin < 0.1): {challenging_cases}/{total_predictions} ({challenging_cases/total_predictions*100:.2f}%)")
    
    logger.info("="*50)
    
    # Save results
    results_to_save = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            results_to_save[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            results_to_save[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            results_to_save[k] = int(v)
        else:
            results_to_save[k] = v
    
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"Results saved to {RESULTS_PATH}")
    
    # Create visualizations
    visualize_results(results, PLOTS_PATH)
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()