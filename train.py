import logging

import pandas as pd
from nlgeval import _strip
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

def getListRouge(hyp_list, refs):
    ref_list = []
    ref_list.append(refs)
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)
    ret_scores = {}
    scorers = [
        (Rouge(), "ROUGE_L")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                # print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            # print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
    del scorers
    return ret_scores['ROUGE_L']

from bart_model import BartModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv('data/train.csv').dropna()
eval_df = pd.read_csv('data/valid.csv').dropna()
test_df = pd.read_csv('data/test.csv').dropna()

train_df.columns = ["input_text", "target_text"]
eval_df.columns = ["input_text", "target_text"]
test_df.columns = ["input_text", "target_text"]

model_args = {
    "overwrite_output_dir": True,
    "train_batch_size": 1,
    "num_train_epochs": 30,
    "max_seq_length":32,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    # "silent": True,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation":False,
    "save_best_model": True,
    # "learning_rate":1e-3,
    "top_k":5,
    "top_p":0.95,
    "max_length": 32,
    "use_early_stopping":True,
    "length_penalty":1.2,
    "best_model_dir":'result/best_model',
    "output_dir":'result',
    "early_stopping_metric": 'Rouge',
    "early_stopping_metric_minimize": False,
}


model = BartModel(pretrained_model=None,args=model_args, model_config='config.json', vocab_file="./tokenize")
#
model.train_model(train_df, eval_data=eval_df, Rouge=getListRouge)