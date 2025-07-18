#!/usr/bin/env python
# coding: utf-8

from datasets import DatasetDict
from modular.data_setup import triplet_collate_fn,TripletDataset,mean_pooling
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import logging
import gc

# Load and split data
ds= DatasetDict.from_csv("data/r_depression_posts.csv")
ds =ds.remove_columns("Unnamed: 0")

train_val, test = ds.train_test_split(test_size=0.03, seed=42).values()
val_size = 0.17 / (1 - 0.03)  
train, val = train_val.train_test_split(test_size=val_size, seed=42).values()

ds_splits = DatasetDict({
    'train': train,
    'validation': val,
    'test': test
})

# Model setup
checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# MEMORY OPTIMIZATION 1: Reduce batch size significantly
BATCH_SIZE = 8  # Reduced from 32
GRADIENT_ACCUMULATION_STEPS = 4  # To maintain effective batch size of 32

train_dataset = TripletDataset(ds_splits["train"], tokenizer)
val_dataset = TripletDataset(ds_splits["validation"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=triplet_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=triplet_collate_fn)

def calculate_triplet_accuracy(anchor_emb, positive_emb, negative_emb):
    pos_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=1)
    neg_sim = F.cosine_similarity(anchor_emb, negative_emb, dim=1)
    accuracy = (pos_sim > neg_sim).float().mean()
    return accuracy.item()

# MEMORY OPTIMIZATION 2: Efficient batch processing function
def process_triplet_batch(model, batch):
    """Process triplet batch efficiently by concatenating inputs"""
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
    with torch.cuda.amp.autocast():  # Mixed precision
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
        embeddings = mean_pooling(outputs, all_attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Split embeddings back into anchor, positive, negative
    batch_size = batch['anchor_input_ids'].size(0)
    anchor_emb = embeddings[:batch_size]
    positive_emb = embeddings[batch_size:2*batch_size]
    negative_emb = embeddings[2*batch_size:]
    
    return anchor_emb, positive_emb, negative_emb

# MEMORY OPTIMIZATION 3: Efficient validation with checkpointing
def validate_model(model, val_loader, triplet_loss, accelerator):
    """Memory-efficient validation function"""
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
                torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy

# Setup training components
triplet_loss = torch.nn.TripletMarginLoss(margin=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# MEMORY OPTIMIZATION 4: Use gradient scaler for mixed precision
scaler = torch.cuda.amp.GradScaler()

accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision='fp16'  # Enable mixed precision
)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# Training parameters
num_epochs = 3
num_update_steps_per_epoch = len(train_loader)
num_training_steps = num_epochs * num_update_steps_per_epoch

logging_steps = 50
eval_steps = 500

from transformers import get_scheduler
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MEMORY OPTIMIZATION 5: Optimized training loop
model.train()
global_step = 0
best_val_accuracy = 0

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_local_main_process)
    
    for batch_idx, batch in enumerate(progress_bar):
        model.train()
        
        # Process batch efficiently with single forward pass
        anchor_emb, positive_emb, negative_emb = process_triplet_batch(model, batch)
        
        # Calculate loss
        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        
        # Scale loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Calculate accuracy for monitoring
        with torch.no_grad():
            accuracy = calculate_triplet_accuracy(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass with mixed precision
        accelerator.backward(loss)
        
        # Update metrics
        epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        epoch_accuracy += accuracy
        num_batches += 1
        
        # Gradient accumulation step
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Clear cache every few steps
            if global_step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
            'acc': f'{accuracy:.4f}',
            'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        })
        
        # Logging
        if global_step % logging_steps == 0:
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_accuracy / num_batches
            logger.info(f"Step {global_step} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        # Validation
        if global_step % eval_steps == 0:
            # Force garbage collection before validation
            torch.cuda.empty_cache()
            gc.collect()
            
            val_loss, val_accuracy = validate_model(model, val_loader, triplet_loss, accelerator)
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if accelerator.is_main_process:
                    accelerator.save_state(f"best_model_checkpoint")
                    logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")
            
            model.train()
    
    # End of epoch validation
    torch.cuda.empty_cache()
    gc.collect()
    
    val_loss, val_accuracy = validate_model(model, val_loader, triplet_loss, accelerator)
    avg_train_loss = epoch_loss / num_batches
    avg_train_accuracy = epoch_accuracy / num_batches
    
    logger.info(f"Epoch {epoch + 1} completed:")
    logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    logger.info(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Save checkpoint at end of epoch
    if accelerator.is_main_process:
        accelerator.save_state(f"epoch_{epoch + 1}_checkpoint")

# Final model save
if accelerator.is_main_process:
    accelerator.save_state("final_model_checkpoint")
    logger.info("Training completed and final model saved!")

# Optional: Save the model in HuggingFace format
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("fine_tuned_sentence_transformer")
    tokenizer.save_pretrained("fine_tuned_sentence_transformer")
    logger.info("Model saved in HuggingFace format!")