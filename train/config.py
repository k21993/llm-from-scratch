import torch

class Config:
    # -----------------------------
    # Dataset paths
    # -----------------------------
    # These should point to where your np.memmap tokenized files are stored.
    # e.g. tiny_stories_tokenized_path_train/train.dat and val.dat exist
    tiny_stories_tokenized_path = "data/tokenized_dataset/tiny_stories/"

    # -----------------------------
    # Model hyperparameters
    # -----------------------------
    vocab_size = 7257        # wc -l tokenizer/.tokenizer_info/vocab.json
    d_model = 16            # small model to verify training works
    num_heads = 4            # must divide d_model evenly
    n_layers = 2             # a couple of transformer blocks only
    seq_len = 32             # short sequence for speed

    # -----------------------------
    # Training hyperparameters
    # -----------------------------
    batch_size = 1
    num_epochs = 3            # just enough to see loss drop
    lr = 3e-4
    beta1 = 0.9
    beta2 = 0.999
    pad_token_id = 0
    num_workers = 0           # for DataLoader

    # -----------------------------
    # Device setup
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Misc
    # -----------------------------
    log_every = 10            # print every n steps
