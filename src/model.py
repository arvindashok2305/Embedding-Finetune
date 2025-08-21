from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

def load_pretrained_model(model_name="BAAI/bge-base-en-v1.5"):
    return SentenceTransformer(model_name)

def setup_training(train_examples, model, batch_size=32):
    dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    return dataloader, train_loss
