from torch.utils.data import Dataset
import torch


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



def triplet_collate_fn(batch):
    return {
        'anchor_input_ids': torch.stack([item['anchor_input_ids'] for item in batch]),
        'anchor_attention_mask': torch.stack([item['anchor_attention_mask'] for item in batch]),
        'positive_input_ids': torch.stack([item['positive_input_ids'] for item in batch]),
        'positive_attention_mask': torch.stack([item['positive_attention_mask'] for item in batch]),
        'negative_input_ids': torch.stack([item['negative_input_ids'] for item in batch]),
        'negative_attention_mask': torch.stack([item['negative_attention_mask'] for item in batch]),
    }



#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
