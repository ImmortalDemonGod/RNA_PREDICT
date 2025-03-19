from rna_predict.pipeline.stageA.stageA_rfold import StageARFoldPredictor

def run_stageA(rna_sequence, predictor):
    """
    Run Stage A prediction on the given RNA sequence using the provided predictor.
    Returns the predicted adjacency matrix (N x N).
    """
    adjacency = predictor.predict_adjacency(rna_sequence)
    return adjacency