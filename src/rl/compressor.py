import torch
from copy import deepcopy
from src.net_wrapper import TorchNetworkWrapper


class PruningQuantizationCompressor(TorchNetworkWrapper):
    def __init__(self, model_args, data_loaders):
        super().__init(model_args)

        self.original_model = deepcopy(self.model)
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
 
        if self.model.is_image_classifier:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.model.device)
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.01, weight_decay=1e-4)

    def reset(self):
        self.model = self.deepcopy(self.original_model)

    def prune_and_quantize(self, pruning_ratio, quant_bits):
        self.prune(pruning_ratio)
        self.quantize(quant_bits)

    def prune(self, pruning_ratio):
        pass

    def quantize(self, quant_bits):
        pass

    def translate_pruning_action(self, pruning_action):
        pruning_action = pruning_action * (self.args.pruning_high - self.args.pruning_low) + self.args.pruning_low
        return np.round(pruning_action, 2).astype(float)

    def translate_quant_action(self, quant_action):
        quant_action = quant_action * (self.args.quant_high - self.args.quant_low) + self.args.quant_low
        return int(np.round(quant_action, 0))

    def train(self, epochs):
        super().train(epochs, self.train_loader, self.criterion, self.optimizer)

    def validate(self):
        super().validate(self.valid_loader, self.criterion)

    def test(self):
        super().test(self.test_loader, self.criterion)

