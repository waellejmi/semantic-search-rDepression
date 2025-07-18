#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import DatasetDict
from modular.data_setup import mean_pooling
from accelerate import Accelerator
import argparse
import logging
from typing import List, Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostSimilarityFinder:
    def __init__(self, model_path: str, data_path: str):
        """
        Initialize the similarity finder
        
        Args:
            model_path: Path to the trained model checkpoint
            data_path: Path to the CSV dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load the dataset
        self.load_dataset(data_path)
        
        # Load model and tokenizer
        self.load_model(model_path)
        
        # Precompute embeddings for all posts
        self.precompute_embeddings()
        
        logger.info("PostSimilarityFinder initialized successfully!")
    
    def load_dataset(self, data_path: str):
        """Load and prepare the dataset"""
        logger.info(f"Loading dataset from {data_path}")
        
        # Load dataset
        ds = DatasetDict.from_csv(data_path)
        
        # Remove unnamed column if exists
        if "Unnamed: 0" in ds.column_names:
            ds = ds.remove_columns("Unnamed: 0")
        
        # Convert to pandas for easier manipulation
        self.df = ds.to_pandas()
        
        # Ensure we have the required columns
        required_columns = ['text', 'url']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(self.df.columns)}")
            
            # If URL column is missing, create a placeholder
            if 'url' not in self.df.columns:
                logger.warning("URL column not found. Creating placeholder URLs...")
                self.df['url'] = [f"https://reddit.com/post_{i}" for i in range(len(self.df))]
        
        logger.info(f"Dataset loaded with {len(self.df)} posts")
        logger.info(f"Columns: {list(self.df.columns)}")
    
    def load_model(self, model_path: str):
        """Load the trained model"""
        logger.info(f"Loading model from {model_path}")
        
        # Initialize accelerator
        accelerator = Accelerator()
        
        # Load base model and tokenizer
        checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)
        
        # Load trained weights
        accelerator.load_state(model_path)
        
        # Prepare model
        self.model = accelerator.prepare(self.model)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
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
    
    def precompute_embeddings(self):
        """Precompute embeddings for all posts in the dataset"""
        logger.info("Precomputing embeddings for all posts...")
        
        # Get all texts
        all_texts = self.df['text'].tolist()
        
        # Compute embeddings in batches
        self.post_embeddings = self.get_embeddings(all_texts, batch_size=32)
        
        logger.info(f"Precomputed embeddings shape: {self.post_embeddings.shape}")
    
    def find_similar_posts(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Find the most similar posts to the query text
        
        Args:
            query_text: The input text to find similar posts for
            top_k: Number of top similar posts to return
            
        Returns:
            List of dictionaries containing post info and similarity scores
        """
        logger.info(f"Finding top {top_k} similar posts for query...")
        
        # Get embedding for query text
        query_embedding = self.get_embeddings([query_text], batch_size=1)
        
        # Calculate similarities with all posts
        similarities = np.dot(self.post_embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for i, idx in enumerate(top_indices):
            similarity_score = similarities[idx]
            
            result = {
                'rank': i + 1,
                'similarity_score': float(similarity_score),
                'text': self.df.iloc[idx]['text'],
                'url': self.df.iloc[idx]['url'],
                'post_index': int(idx)
            }
            results.append(result)
        
        return results
    
    def print_results(self, query_text: str, results: List[Dict]):
        """Pretty print the results"""
        print("\n" + "="*80)
        print("QUERY TEXT:")
        print("="*80)
        print(f"{query_text}")
        print("\n" + "="*80)
        print("TOP 3 SIMILAR POSTS:")
        print("="*80)
        
        for result in results:
            print(f"\n🏆 RANK {result['rank']} (Similarity: {result['similarity_score']:.4f})")
            print("-" * 60)
            print(f"📝 TEXT: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
            print(f"🔗 URL: {result['url']}")
            print(f"📊 INDEX: {result['post_index']}")
            print("-" * 60)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("\n🔍 Interactive Post Similarity Finder")
        print("="*50)
        print("Enter your text to find similar posts (or 'quit' to exit)")
        
        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("⚠️ Please enter a valid query.")
                    continue
                
                # Find similar posts
                results = self.find_similar_posts(query, top_k=3)
                
                # Print results
                self.print_results(query, results)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Find similar posts using trained model')
    parser.add_argument('--model_path', type=str, default='best_model_checkpoint',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/r_depression_posts.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--query', type=str, default=None,
                       help='Single query text (optional)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top similar posts to return')
    
    args = parser.parse_args()
    
    try:
        # Initialize the similarity finder
        finder = PostSimilarityFinder(args.model_path, args.data_path)
        
        if args.query:
            # Single query mode
            results = finder.find_similar_posts(args.query, top_k=args.top_k)
            finder.print_results(args.query, results)
        else:
            # Interactive mode
            finder.interactive_mode()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()


# Example usage functions for Jupyter notebook or direct import
def quick_find_similar(model_path: str, data_path: str, query_text: str, top_k: int = 3):
    """
    Quick function to find similar posts
    
    Args:
        model_path: Path to trained model
        data_path: Path to dataset
        query_text: Text to find similar posts for
        top_k: Number of results to return
        
    Returns:
        List of similar posts
    """
    finder = PostSimilarityFinder(model_path, data_path)
    results = finder.find_similar_posts(query_text, top_k)
    finder.print_results(query_text, results)
    return results


# Example usage in script
if __name__ == "__main__":
    # You can also use it directly like this:
    # 
    # finder = PostSimilarityFinder("best_model_checkpoint", "data/r_depression_posts.csv")
    # results = finder.find_similar_posts("I'm feeling really depressed today", top_k=3)
    # finder.print_results("I'm feeling really depressed today", results)
    
    main()