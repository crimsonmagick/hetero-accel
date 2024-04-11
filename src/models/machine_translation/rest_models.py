import torchtext


def t5(pretrained=True):
    """T5 Model for summarization, sentiment classification, and translation tasks
    """
    model = torchtext.models.T5_BASE_GENERATION.get_model()
    return model
