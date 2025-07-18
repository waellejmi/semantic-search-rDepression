from torch.utils.data import Dataset
from datasets import Dataset  as HFDataset
import torch

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_length=512):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        anchor = self.tokenizer(
            triplet['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive = self.tokenizer(
            triplet['positive'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative = self.tokenizer(
            triplet['negative'], 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor['input_ids'].squeeze(),
            'anchor_attention_mask': anchor['attention_mask'].squeeze(),
            'positive_input_ids': positive['input_ids'].squeeze(),
            'positive_attention_mask': positive['attention_mask'].squeeze(),
            'negative_input_ids': negative['input_ids'].squeeze(),
            'negative_attention_mask': negative['attention_mask'].squeeze(),
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