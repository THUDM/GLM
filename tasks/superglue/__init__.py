default_metrics = {
    "record": [("EM", qa_exact_match), ("F1", qa_f1)],
    "copa": [("accuracy", accuracy_metric)],
    "rte": [("accuracy", accuracy_metric)],
    "boolq": [("accuracy", accuracy_metric)],
    "wic": [("accuracy", accuracy_metric)],
    'wsc': [("accuracy", accuracy_metric)],
    "cb": [("accuracy", accuracy_metric), ("f1-macro", f1_macro_metric)],
    "multirc": [("f1a", f1_metric), ("em", multirc_em), ("acc", accuracy_metric)]
}