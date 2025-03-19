import torch
from rna_predict.pipeline.stageA.rfold import StageARFoldPredictor 

config = {"num_hidden": 128, "dropout": 0.3}
checkpoint = "./checkpoints/RNAStralign_trainset_pretrained.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

predictor = StageARFoldPredictor(config, checkpoint_path=checkpoint, device=device)
adj = predictor.predict_adjacency("AAUGCGUCU...")
print("Adjacency shape:", adj.shape)
