import logging


logger = logging.getLogger(__name__)


class MultiDNNWorkload:
    def __init__(self, dnn_dict, dataset_dict, print_frequency_dict):
        self.dnns = dnn_dict
        self.datasets = dataset_dict
        self.print_frequency = print_frequency_dict

    def get_data_loaders(self, arch):
        return self.datasets[self.dnns[arch].dataset]

    def get_summary(self, arch):
        return self.dnns[arch].summary

    def get_layer_dimensions(self, arch, layer_name):
        return self.dnns[arch].summary[layer_name].dimensions

    def get_layer_type(self, arch, layer_name):
        return self.dnns[arch].summary[layer_name].layer_type

