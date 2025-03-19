from rna_predict.stageA_rfold import StageARFoldPredictor

def run_stageA(rna_sequence, predictor):
    """
    Run Stage A prediction on the given RNA sequence using the provided predictor.
    Returns the predicted adjacency matrix.
    """
    return predictor.predict_adjacency(rna_sequence)
