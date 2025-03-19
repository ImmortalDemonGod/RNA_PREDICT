import torch
# from rna_predict.pipeline/stageA import StageARFoldPredictor

config = {"num_hidden": 128, "dropout": 0.3}
checkpoint = "./checkpoints/RNAStralign_trainset_pretrained.pth"
device = torch.device("cuda:0")

predictor = StageARFoldPredictor(config, checkpoint_path=checkpoint, device=device)
adj = predictor.predict_adjacency("AAUGCGUCU...")
print("Adjacency shape:", adj.shape)
