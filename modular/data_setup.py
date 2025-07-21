from torch.utils.data import Dataset
from datasets import DatasetDict
import torch
import random


def get_train_test_val_dataset(path):
    ds= DatasetDict.from_csv(path)

    train_val, test = ds.train_test_split(test_size=0.03, seed=42).values() #3% test dataset
    val_size = 0.17 / (1 - 0.03)  # ensure 17% of total dataset
    train, val = train_val.train_test_split(test_size=val_size, seed=42).values() # 17 val datadset

    ds_splits = DatasetDict({
        'train': train,
        'validation': val,
        'test': test
    })

    return ds_splits



def build_ranking_test_dataset(test_dataset, num_negatives=5):
    """
    Constructs ranking evaluation examples per anchor with 1 positive + N random negatives.
    """
    examples = []
    all_negatives = test_dataset['negative']

    for anchor, pos in zip(test_dataset['text'], test_dataset['positive']):
        # Sample k negatives randomly (excluding true positive/neg)
        negs = set()
        while len(negs) < num_negatives:
            candidate = random.choice(all_negatives)
            if candidate != pos:
                negs.add(candidate)
        candidates = [pos] + list(negs)
        random.shuffle(candidates)
        label = candidates.index(pos)
        examples.append({
            'anchor': anchor,
            'candidates': candidates,
            'label': label
        })
    return examples



def triplet_collate_fn(batch):
    return {
        'anchor_input_ids': torch.stack([item['anchor_input_ids'] for item in batch]),
        'anchor_attention_mask': torch.stack([item['anchor_attention_mask'] for item in batch]),
        'positive_input_ids': torch.stack([item['positive_input_ids'] for item in batch]),
        'positive_attention_mask': torch.stack([item['positive_attention_mask'] for item in batch]),
        'negative_input_ids': torch.stack([item['negative_input_ids'] for item in batch]),
        'negative_attention_mask': torch.stack([item['negative_attention_mask'] for item in batch]),
    }




class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=512):
        self.max_length = max_length
        
        self.encoded_triplets = []
        for triplet in triplets:
            encoded = {
                "anchor": tokenizer(
                    triplet["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ),
                "positive": tokenizer(
                    triplet["positive"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ),
                "negative": tokenizer(
                    triplet["negative"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ),
            }
            self.encoded_triplets.append(encoded)

    def __len__(self):
        return len(self.encoded_triplets)

    def __getitem__(self, idx):
        triplet = self.encoded_triplets[idx]

        return {
            "anchor_input_ids": triplet["anchor"]["input_ids"].squeeze(),
            "anchor_attention_mask": triplet["anchor"]["attention_mask"].squeeze(),
            "positive_input_ids": triplet["positive"]["input_ids"].squeeze(),
            "positive_attention_mask": triplet["positive"]["attention_mask"].squeeze(),
            "negative_input_ids": triplet["negative"]["input_ids"].squeeze(),
            "negative_attention_mask": triplet["negative"]["attention_mask"].squeeze(),
        }

