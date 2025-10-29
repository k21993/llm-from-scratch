from tqdm import tqdm
import torch #type: ignore
from torch.utils.data import DataLoader
from dataset.tiny_stories import TinyStories
from model.llm import TransformerLM
from train.optimizer import Adam
from train.loss import CrossEntropyLoss
from train.utils import get_trainable_params
from train.config import Config
import logging

#init a logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(config):
    #1. define dataloaders
    train_dataset = TinyStories(
        np_file_path=config.tiny_stories_tokenized_path,
        split="val",
        seq_len=config.seq_len,
        pad_token_id=config.pad_token_id
        )
    
    # val_dataset = TinyStories(
    #     np_file_path=config.tiny_stories_tokenized_path,
    #     split="val",
    #     seq_len=config.seq_len,
    #     pad_token_id=config.pad_token_id
    #     )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=(config.device == "cuda"),
        num_workers=config.num_workers,
        shuffle=True
        )
    
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     pin_memory=(config.device == "cuda"),
    #     num_workers=config.num_workers,
    #     )
    
    #2. load model
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        seq_len=config.seq_len,
        n_layers=config.n_layers
    )
    model.to(config.device)
    logger.info(model)

    #4. init optim
    trainable_params, num_trainable_params = get_trainable_params(model)
    logger.info(f"num trainable params: {num_trainable_params}")
    optim = Adam(
        params=trainable_params,
        lr=config.lr,
        beta1=config.beta1,
        beta2=config.beta2
    )

    #5. init loss
    ce_loss = CrossEntropyLoss()

    #training loop
    for epoch in range(config.num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for inp, label in pbar:
            inp = inp.to(config.device)
            label = label.to(config.device)
            pred = model(inp)
            loss = ce_loss(y_true=label, y_pred=pred)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix({"loss": loss.item()})


if __name__ == "__main__":
    train(Config)
        
    
