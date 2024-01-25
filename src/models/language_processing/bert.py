import torch
from functools import partial
# from pytorch_transformers import BertConfig, BertForQuestionAnswering, BertTokenizer


class Bert:
    pass


def _bert(model_path, do_lower_case=False):
    config = BertConfig.from_pretrained(model_path + "/bert_config.json")
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
    return model, tokenizer


def bert():
    raise NotImplementedError
