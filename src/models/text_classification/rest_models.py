import torchtext

def xlmr(num_classes=2, input_dim=768, pretrained=True):
    """XLM-Roberta model
    """
    classifier_head = torchtext.models.RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
    model = torchtext.models.XLMR_BASE_ENCODER.get_model(head=classifier_head)
    return model


def t5(pretrained=True):
    """T5 Model for summarization, sentiment classification, and translation tasks
    """
    model = torchtext.models.T5_BASE_GENERATION.get_model()
    return model
