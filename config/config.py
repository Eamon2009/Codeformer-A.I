import torch

# # Data
# data_path    = "traindata.txt"
cleaned_path = "cleaned.txt"
train_split  = 0.9

# Model
vocab_size   = None   # set automatically after data is loaded

# Training
block_size   = 18
batch_size   = 4
seed         = 1337

# System
device = "cuda" if torch.cuda.is_available() else "cpu"