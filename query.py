import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import argparse
import gradio as gr


from modular.engine import mean_pooling
from modular.utils import set_up_info_loggger

logger = set_up_info_loggger(name="Query")


class PostSimilarityFinder:
    def __init__(self, model_path: str, dataset_dir: str, faiss_index_path: str):
        """
        Args:
            model_path: Directory with unwrapped fine-tuned model
            dataset_dir: Path to dataset with precomputed embeddings
            faiss_index_path: Path to saved FAISS index
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.load_model(model_path)
        self.load_indexed_dataset(dataset_dir, faiss_index_path)

    def load_model(self, model_path: str):
        logger.info(f"Loading model and tokenizer from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def load_indexed_dataset(self, dataset_dir: str, faiss_index_path: str):
        logger.info(f"Loading dataset from {dataset_dir}")
        self.dataset = Dataset.load_from_disk(dataset_dir)
        self.dataset.load_faiss_index("embeddings", faiss_index_path)
        logger.info(f"Dataset loaded with {len(self.dataset)} entries")

    def get_query_embedding(self, query: str) -> np.ndarray:
        encoded = self.tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            embedding = mean_pooling(output, encoded["attention_mask"])
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()[0]

    def find_similar_posts(self, query_text: str, top_k: int = 3):
        logger.info(f"Searching for top {top_k} posts similar to query...")
        query_embedding = self.get_query_embedding(query_text)
        scores, examples = self.dataset.get_nearest_examples(
            "embeddings", query_embedding, k=top_k
        )

        zipped = list(zip(scores, examples["text"], examples.get("url", [""] * top_k)))
        zipped.sort(key=lambda x: x[0], reverse=True)

        results = []
        for i, (score, text, url) in enumerate(zipped):
            results.append(
                {
                    "rank": i + 1,
                    "similarity_score": float(score),
                    "text": text,
                    "url": url,
                }
            )

        return results

    def print_results(self, query_text: str, results):
        print("\n" + "=" * 80)
        print("QUERY TEXT:")
        print("=" * 80)
        print(f"{query_text}")
        print("\n" + "=" * 80)
        print(f"TOP {len(results)} SIMILAR POSTS:")
        print("=" * 80)

        for result in results:
            print(
                f"\nSimilar Post {result['rank']} (Similarity: {result['similarity_score']:.4f})"
            )
            print("-" * 60)
            preview = (
                result["text"][:200] + "..."
                if len(result["text"]) > 200
                else result["text"]
            )
            print(f"Text: {preview}")
            print(f"URL: {result.get('url', '')}")
            print("-" * 60)

    def interactive_mode(self, top_k=3):
        print("\nInteractive Post Similarity Finder")
        print("=" * 50)
        print("Enter your query to find similar posts (type 'quit' to exit).")

        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("Exiting. Goodbye.")
                    break
                if not query:
                    print("Please enter a non-empty query.")
                    continue
                results = self.find_similar_posts(query, top_k=top_k)
                self.print_results(query, results)
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting.")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")


def launch_gradio_interface(finder: PostSimilarityFinder, top_k: int = 3):
    def query_interface(user_query):
        results = finder.find_similar_posts(user_query, top_k=top_k)
        return "\n\n".join(
            [
                f"### Post {r['rank']} (Score: {r['similarity_score']:.4f})\n"
                f"{r['text'][:500]}{'...' if len(r['text']) > 500 else ''}\n"
                f"[Link]({r['url']})"
                if r["url"]
                else ""
                for r in results
            ]
        )

    gr.Interface(
        fn=query_interface,
        inputs=gr.Textbox(
            label="Enter your query", placeholder="e.g. I feel empty...", lines=3
        ),
        outputs=gr.Markdown(label="Top Matching Reddit Posts"),
        title="Semantic Search on r/depression Posts",
        description="Find similar Reddit posts using a fine-tuned MiniLM model and FAISS.",
        flagging_mode="never",
    ).launch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/all-MiniLM-L6-v2-finetuned-on-rDepression",
    )
    parser.add_argument("--dataset_dir", type=str, default="data/with_embeddings")
    parser.add_argument(
        "--faiss_index_path", type=str, default="data/with_embeddings/faiss.index"
    )
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    finder = PostSimilarityFinder(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        faiss_index_path=args.faiss_index_path,
    )

    if args.query:
        results = finder.find_similar_posts(args.query, args.top_k)
        finder.print_results(args.query, results)
    else:
        use_gradio = input("Launch Gradio interface? (y/N): ").lower() == "y"
        if use_gradio:
            launch_gradio_interface(finder, top_k=args.top_k)
        else:
            finder.interactive_mode(top_k=args.top_k)


if __name__ == "__main__":
    main()
