import matplotlib.pyplot as plt
import logging
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel




def set_up_info_loggger(name=__name__):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)
    return logger

def visualize_results(results,logger, save_path=None):
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
    
