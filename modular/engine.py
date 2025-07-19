import torch.nn.functional as F
from tqdm import tqdm
import torch
import gc


def calculate_triplet_accuracy(anchor_emb, positive_emb, negative_emb):
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
    neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1)
    accuracy = (pos_sim > neg_sim).float().mean()
    return accuracy.item()


#Mean Pooling - Take attention mask into account for correct averaging (copied from hf sentence-transformers/all-MiniLM-L6-v2')
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



# MEMORY OPTIMIZATION 1: Efficient batch processing function
def process_triplet_batch(model, batch):
    # Concatenate all inputs for single forward pass
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
    
    # Single forward pass for all triplets
    with torch.cuda.amp.autocast('cuda'):  # Mixed precision
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
        embeddings = mean_pooling(outputs, all_attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Split embeddings back into anchor, positive, negative
    batch_size = batch['anchor_input_ids'].size(0)
    anchor_emb = embeddings[:batch_size]
    positive_emb = embeddings[batch_size:2*batch_size]
    negative_emb = embeddings[2*batch_size:]
    
    return anchor_emb, positive_emb, negative_emb



# MEMORY OPTIMIZATION 2: Efficient validation 
def validate_model(model, val_loader, triplet_loss, accelerator):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", disable=not accelerator.is_local_main_process):
            # Process batch efficiently
            anchor_emb, positive_emb, negative_emb = process_triplet_batch(model, batch)

            # Calculate loss
            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            accuracy = calculate_triplet_accuracy(anchor_emb, positive_emb, negative_emb)

            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Clear cache periodically
            if num_batches % 10 == 0:
                clear_cuda_cache()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy




# MEMORY OPTIMIZATION 3: we use it for mixed precision but we got Accelarte from HF already doing it 
scaler = torch.cuda.amp.GradScaler('cuda')


# MEMORY OPTIMIZATION 4 : Clear GPU VRAM cache + force garbage colletion   ( use it every few steps ) 
def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
