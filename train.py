#!/usr/bin/env python
# coding: utf-8

from modular.data_setup import triplet_collate_fn,TripletDataset,get_train_test_val_dataset
from modular.engine import calculate_triplet_accuracy,validate_model,process_triplet_batch,clear_cuda_cache
from modular.utils import set_up_info_loggger
from transformers import AutoTokenizer, AutoModel, get_scheduler

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

logger = set_up_info_loggger()


#g Get Data
ds_splits = get_train_test_val_dataset("data/r_depression_posts.csv")

# Model setup
checkpoint = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# MEMORY OPTIMIZATION 1: Reduce batch size significantly
BATCH_SIZE = 8  
GRADIENT_ACCUMULATION_STEPS = 4  # To maintain effective batch size of 32

train_dataset = TripletDataset(ds_splits["train"], tokenizer)
val_dataset = TripletDataset(ds_splits["validation"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=triplet_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=triplet_collate_fn)



# Setup training components
triplet_loss = torch.nn.TripletMarginLoss(margin=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)



accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    mixed_precision='fp16' 
)

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# Training parameters
num_epochs = 3
num_update_steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION_STEPS
num_training_steps = num_epochs * num_update_steps_per_epoch
num_warmup_steps = int(0.05 * num_training_steps)

# Monotoring parameters
logging_steps = 50
eval_steps = 500

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)


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
                clear_cuda_cache()
        
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
        if (global_step % eval_steps == 0 and (batch_idx + 1) < len(train_loader)):
            clear_cuda_cache()
            
            val_loss, val_accuracy = validate_model(model, val_loader, triplet_loss, accelerator)
            logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if accelerator.is_main_process:
                    accelerator.save_state(f"models/best_model_checkpoint")
                    logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")
            
            model.train()
    
    # End of epoch validation
    clear_cuda_cache()
    
    val_loss, val_accuracy = validate_model(model, val_loader, triplet_loss, accelerator)
    avg_train_loss = epoch_loss / num_batches
    avg_train_accuracy = epoch_accuracy / num_batches
    
    logger.info(f"Epoch {epoch + 1} completed:")
    logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    

#  Save the model 
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(model)
    save_path = "models/all-MiniLM-L6-v2-finetuned-rDepression"
    unwrapped_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")