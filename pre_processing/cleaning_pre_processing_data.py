#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_all= pd.read_json('../data/_reddit-posts-gatherer-en.submissions_subset.json')
df_all.info()


# In[69]:


df = df_all[['body', 'url', 'title']].copy()

df['body'] = (
    df['body']
    .str.replace('\n', '. ', regex=False)               
    .str.replace(r'\s+', ' ', regex=True)                
    .str.replace(r'\.\s*\.', '.', regex=True)             
    .str.strip()                                            
)

df["text"] = df["title"].str.strip() + " " + df["body"].str.strip()
df = df[["text", "url"]]

df = df.dropna()
# df = df.drop_duplicates()

df.info()


# In[70]:


df


# In[71]:


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = df['text'].to_list()

embeddings = model.encode(sentences,
                          batch_size=64)
print(embeddings.shape)


# In[72]:


print("Max Sequence Length:", model.max_seq_length)


# In[73]:


similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)


# In[ ]:


import random
import torch
import numpy as np

def mine_triplets_simple(similarities, sentences, margin=0.3):
    """
    Simple and robust semi-hard triplet mining
    args:
        -similarities: matrix of similatriy of the sentences embeddings
        -sentences: list of sentences/ pargraphs
        -margin: margin off semmi hard mining (how close the negative to the postive) 

    """
    triplets = []
    n_samples = len(sentences)

    for anchor_idx in range(n_samples):
        sims = similarities[anchor_idx].clone()
        sims[anchor_idx] = -1  # Exclude self

        # Find positive: closest to thje  sample embedding
        pos_idx = sims.argmax().item()
        pos_sim = sims[pos_idx].item()


        mask = (sims < pos_sim) & (sims > pos_sim - margin) & (torch.arange(len(sims)) != pos_idx)
        candidate_neg_indices = torch.where(mask)[0]

        if len(candidate_neg_indices) == 0:
            # Fallback to hardest negative (closest among negatives)
            remaining_mask = torch.arange(len(sims)) != pos_idx
            remaining_indices = torch.where(remaining_mask)[0]
            if len(remaining_indices) > 0:
                neg_idx = remaining_indices[sims[remaining_indices].argmax()].item()
            else:
                continue
        else:
            neg_idx = random.choice(candidate_neg_indices.tolist())


        triplets.append({
            "anchor_idx": anchor_idx,
            "positive_idx": pos_idx,
            "negative_idx": neg_idx,
            "anchor": sentences[anchor_idx],
            "positive": sentences[pos_idx],
            "negative": sentences[neg_idx],
            "pos_sim": pos_sim,
            "neg_sim": sims[neg_idx].item(),
            "margin_violation": pos_sim - sims[neg_idx].item() < margin
        })

    return triplets

triplets = mine_triplets_simple(similarities, sentences, margin=0.3)

print(f"Generated {len(triplets)} triplets")
print(f"Margin violations: {sum(t['margin_violation'] for t in triplets)}")


pos_sims = [t['pos_sim'] for t in triplets]
neg_sims = [t['neg_sim'] for t in triplets]
margins = [t['pos_sim'] - t['neg_sim'] for t in triplets]

print(f"\nStatistics:")
print(f"Positive similarities: {np.mean(pos_sims):.4f} ± {np.std(pos_sims):.4f}")
print(f"Negative similarities: {np.mean(neg_sims):.4f} ± {np.std(neg_sims):.4f}")
print(f"Actual margins: {np.mean(margins):.4f} ± {np.std(margins):.4f}")


# In[75]:


triplet_df = pd.DataFrame.from_dict(triplets)
triplet_df


# In[89]:


only_triple_df=triplet_df[['anchor','positive','negative']].copy()
only_triple_df


# In[90]:


print(len(df), len(only_triple_df),len(triplet_df))


# In[91]:


final_df = df.merge(only_triple_df, left_index=True, right_index=True, how='left')
final_df


# In[80]:


final_df.shape


# In[81]:


final_df.isna().sum()


# In[92]:


final_df.drop(columns=['anchor'],inplace=True)
final_df


# In[ ]:


final_df.to_csv('../data/r_depression_posts.csv',index=False)

