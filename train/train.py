
import torch #type: ignore
from torch.utils.data import DataLoader
from dataset.tiny_stories import TinyStories
from model.llm import TransformerLM
from optimizer import Adam

def train(config):
    #1. define dataloaders
    train_dataset = TinyStories(
        np_file_path=config.tiny_stories_tokenized_path_train,
        split="train",
        seq_len=config.seq_len,
        pad_token_id=config.pad_token_id
        )
    
    val_dataset = TinyStories(
        np_file_path=config.tiny_stories_tokenized_path_val,
        split="val",
        seq_len=config.seq_len,
        pad_token_id=config.pad_token_id
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
        device=config.device,
        shuffle=True
        )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
        device=config.device
        )
    
    #2. load model
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        n_layers=config.n_layers
    )

    model.to(config.device)

    #init optim
    optim = Adam(
        params=params,
        lr=config.lr,
        beta1=config.beta1,
        beta2=config.beta2
    )
    
