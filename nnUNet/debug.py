from nnunet.evaluation.evaluator import evaluate_folder


if __name__ == "__main__":
    ref = "/raid/ai_biomed/test_ref"
    pred = "/raid/ai_biomed/test_pred"
    l = (2,)
    r = evaluate_folder(ref, pred, l)