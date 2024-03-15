import logging
import os.path
import shutil
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.models import segmentation as seg
from torchvision.models import detection as det
import torchvision.datasets as datasets
from functools import partial
from src.dataset_utils import VOCDetTransform, get_voc_seg_transform, get_coco_transform


logger = logging.getLogger(__name__)


def load_data(dataset, dataset_path, arch, batch_size, workers,
              validation_split, train_size, valid_size, test_size, test_only,
              verbose):

    dataset_fn = __dataset_factory(dataset, batch_size=batch_size)
    train_loader, valid_loader, test_loader = get_data_loaders(dataset_fn, dataset_path, arch, batch_size, workers,
                                                               validation_split, train_size, valid_size, test_size,
                                                               test_only)
    if verbose:
        if test_only:
            logger.info('Dataset sizes:\n\ttest={}'.format(len(test_loader.sampler)))
        else:
            logger.info('Dataset sizes:\n\ttraining={}\n\tvalidation={}\n\ttest={}'.format(
                    len(train_loader.sampler), len(valid_loader.sampler), len(test_loader.sampler)
                )
            )
    return train_loader, valid_loader, test_loader


def __dataset_factory(dataset, batch_size):
    try:
        return {
            # image classification
            'cifar10': get_cifar10_dataset,
            'cifar100': get_cifar100_dataset,
            'imagenet': get_imagenet_dataset,
            'tiny-imagenet': get_tinyimagenet_dataset,
            'mnist': get_mnist_dataset,
            # image segmentation
            'voc_seg': get_vocseg_dataset,
            # object detection
            'voc_det': get_vocdet_dataset,
            'coco': get_coco_dataset,
            # language processing,
            'sst2': partial(get_sst2_dataset, batch_size=batch_size),
            'multi30k': get_multi30k_dataset, 
            # video processing
            'moving_mnist': get_moving_mnist_dataset,
            'kinetics': get_kinetics_dataset,
        }[dataset.lower()]
    except KeyError:
        raise ValueError('Dataset {} not supported'.format(dataset))


def get_data_loaders(dataset_fn, data_dir, arch, batch_size, workers, validation_split,
                     effective_train_size, effective_valid_size, effective_test_size, test_only):
    """Create data loaders of selected size and random sampling
    """
    def split_list(list_to_split, ratio, shuffle=False):
        if shuffle:
            np.random.shuffle(list_to_split)
        split_idx = int(np.floor(ratio * len(list_to_split)))
        return list_to_split[:split_idx], list_to_split[split_idx:]

    # load the datasets
    train_dataset, test_dataset = dataset_fn(data_dir, arch, load_train=not test_only, load_test=True)

    # custom way of not using multiple processes
    if workers == 0:
        torch.set_num_threads(1)

    test_indices = list(range(len(test_dataset)))
    #test_sampler = SwitchingSubsetRandomSampler(test_indices, effective_test_size)
    test_sampler = torch.utils.data.RandomSampler(test_indices,
                                                  num_samples=int(effective_test_size * len(test_indices)))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=workers,
        #   worker_init_fn=worker_init_fn,
            pin_memory=True, drop_last=True
    )
    if test_only:
        return None, None, test_loader

    if train_dataset is None:
        return None, test_loader, test_loader

    train_indices, valid_indices = split_list(list(range(len(train_dataset))), 1 - validation_split, shuffle=True)
    train_indices, valid_indices = list(train_indices), list(valid_indices)

    #train_sampler = SwitchingSubsetRandomSampler(train_indices, effective_train_size)
    train_sampler = torch.utils.data.RandomSampler(train_indices,
                                                   num_samples=int(effective_train_size * len(train_indices)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=workers,
        # worker_init_fn=worker_init_fn,
        pin_memory=True, drop_last=True
    )

    valid_loader = None
    if valid_indices:
        #valid_sampler = SwitchingSubsetRandomSampler(valid_indices, effective_valid_size)
        valid_sampler = torch.utils.data.RandomSampler(valid_indices,
                                                       num_samples=int(effective_valid_size * len(valid_indices)))
        valid_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=workers,
            # worker_init_fn=worker_init_fn,
            pin_memory=True, drop_last=True
        )

    return train_loader, valid_loader or test_loader, test_loader


### Dataset-specific functions 

def get_cifar10_dataset(cifar10_path, arch, load_train, load_test):
    """Load the CIFAR10 dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(
            root=cifar10_path, train=True, download=True, transform=train_transform
        )

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR10(
            root=cifar10_path, train=False, download=True, transform=test_transform
        )

    return train_dataset, test_dataset


def get_cifar100_dataset(cifar100_path, arch, load_train, load_test):
    """Load the CIFAR100 dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR100(
            root=cifar100_path, train=True, download=True, transform=train_transform
        )

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = datasets.CIFAR100(
            root=cifar100_path, train=False, download=True, transform=test_transform
        )

    return train_dataset, test_dataset


def get_imagenet_dataset(data_dir, arch, load_train=True, load_test=True):
    """Load the ImageNet dataset
    """
    if 'inception' in arch.lower():
        resize, crop = 336, 299
    else:
        resize, crop = 256, 224
    if 'googlenet' in arch.lower():
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')

    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ])

        test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def get_tinyimagenet_dataset(data_dir, load_train=True, load_test=True):
    """Load tiny-imagenet 200 dataset"""
    def restructure_tiny_imagenet():
        val_dir = os.path.join(data_dir, 'val')

        # read annotations and assign images to their labels
        val_dict = {}
        with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
            for line in f.readlines():
                image_file, label, *_ = line.split('\t')
                val_dict[image_file] = label

        # put images in their respective label folders (create if they dont exist)
        for img, folder in val_dict.items():
            newpath = os.path.join(val_dir, folder)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_dir, 'images', img)):
                shutil.move(os.path.join(val_dir, 'images', img), os.path.join(newpath, img))

        # remove empty directory of original images
        os.rmdir(os.path.join(val_dir, 'images'))

    if os.path.isdir(os.path.join(data_dir, 'val', 'images')):
        restructure_tiny_imagenet()

    # Define transformation sequence for image pre-processing
    # If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
    # If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406], 
    # std=[0.229, 0.224, 0.225])

    #transform = transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
    preprocess_transform_pretrain = transforms.Compose([
                    transforms.Resize(256), # Resize images to 256 x 256
                    transforms.CenterCrop(224), # Center crop image
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # Converting cropped images to tensors
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    #val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'val')

    train_dataset = None
    if load_train:
        train_dataset = datasets.ImageFolder(train_dir, preprocess_transform_pretrain)

    test_dataset = None
    if load_test:
        test_dataset = datasets.ImageFolder(test_dir, preprocess_transform_pretrain)

    return train_dataset, test_dataset


def get_mnist_dataset(data_dir, arch, load_train=True, load_test=True):
    """Load the MNIST dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True,
                                       download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root=data_dir, train=False,
                                      download=True, transform=test_transform)

    return train_dataset, test_dataset


def get_vocseg_dataset(data_dir, arch, load_train=True, load_test=True):
    """Get the Pascal VOC image segmentation dataset from PyTorch
    """
    traindata = testdata = None

    weights = None
    if arch == 'fcn_resnet50':
        weights = seg.FCN_ResNet50_Weights.DEFAULT
    elif arch == 'fcn_resnet101':
        weights = seg.FCN_ResNet101_Weights.DEFAULT
    elif arch == 'deeplabv3':
        weights = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    elif arch == 'lraspp':
        weights = seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    transforms_both = transforms.ToTensor() if weights is None else get_voc_seg_transform(weights)

    if load_train:
        traindata = datasets.VOCSegmentation(data_dir,
                                             year="2012",
                                             image_set="train",
                                             download=False,
                                             transforms=transforms_both)

    if load_test:
        testdata = datasets.VOCSegmentation(data_dir,
                                            year='2012',
                                            image_set="val",
                                            download=False,
                                            transforms=transforms_both)
    return traindata, testdata


def get_vocdet_dataset(data_dir, arch, load_train=True, load_test=True):
    """Get the Pascal VOC object detection dataset from PyTorch
    """
    traindata = testdata = None
    if load_train:
        traindata = datasets.VOCDetection(data_dir, year="2012",
                                          image_set="train",
                                          download=False,
                                          transforms=VOCDetTransform())

    if load_test:
        testdata = datasets.VOCDetection(data_dir, year='2012',
                                        image_set="val",
                                        download=False,
                                        transforms=VOCDetTransform())
        
    return traindata, testdata


class CocoDetection(datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco_dataset(data_dir, arch, load_train=True, load_test=True):
    """Get the MS Coco object detection dataset from PyTorch
    """
    traindata = testdata = None

    weights = None
    if arch == 'ssd300_vgg16':
        weights = det.SSD300_VGG16_Weights.DEFAULT
    elif arch == 'retinanet_resnet50':
        weights = det.RetinaNet_ResNet50_FPN_Weights.DEFAULT
    elif arch == 'fasterrcnn_resnet50':
        weights = det.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms_both = transforms.ToTensor() if weights is None else get_coco_transform(weights)

    if load_train:
        logger.warning('Training data are currently unavailable for the MS COCO dataset')

    if load_test:
        testdata = CocoDetection(img_folder=os.path.join(data_dir, "val2017"),
                                 ann_file=os.path.join(data_dir, 'annotations', "instances_val2017.json"),
                                 transforms=transforms_both)
    return traindata, testdata


def get_sst2_dataset(data_dir, arch, load_train=True, load_test=True, batch_size=16):
    """Get the SST-2 dataset for binary text classification 
    """
    if 'xlmr' in arch:
        text_transform = torchtext.models.XLMR_BASE_ENCODER.transform()
    
    def batch_transform(x):
        """Transform the raw dataset using API (i.e apply transformation on the whole batch)
        """
        return {"token_ids": text_transform(x["text"]), "target": x["label"]}

    train_dataset = test_dataset = None
    if load_train:
        train_dataset = torchtext.datasets.SST2(root=data_dir, split="train")
        train_dataset = train_dataset.batch(batch_size).rows2columnar(["text", "label"])
        train_dataset = train_dataset.map(lambda x: batch_transform)

    if load_test:
        test_dataset = torchtext.datasets.SST2(root=data_dir, split="test")
        test_dataset = test_dataset.batch(batch_size).rows2columnar(["text", "label"])
        test_dataset = test_dataset.map(lambda x: batch_transform)

    return train_dataset, test_dataset


def get_multi30k_dataset(data_dir, arch, load_train=True, load_test=True, batch_size=5):
    """ Multi30k dataset for English to German translation
    """
    def apply_prefix(task, x):
        return f"{task}: " + x[0], x[1]

    language_pair = ("en", "de")
    task = "translate English to German"

    train_datapipe = test_datapipe = None
    if load_train:
        train_datapipe = torchtext.datasets.Multi30k(split="train", language_pair=language_pair)
        train_datapipe = train_datapipe.map(partial(apply_prefix, task))
        train_datapipe = train_datapipe.batch(batch_size)
        train_datapipe = train_datapipe.rows2columnar(["english", "german"])

    if load_test:
        test_datapipe = torchtext.datasets.Multi30k(split="test", language_pair=language_pair)
        test_datapipe = test_datapipe.map(partial(apply_prefix, task))
        test_datapipe = test_datapipe.batch(batch_size)
        test_datapipe = test_datapipe.rows2columnar(["english", "german"])

    return train_datapipe, test_datapipe


def get_moving_mnist_dataset(data_dir, arch, load_train=True, load_test=True, batch_size=5):
    """MovingMNIST dataset
    """
    train_dataset = test_dataset = None
    if load_train:
        train_dataset = datasets.MovingMNIST(root=data_dir,
                                             download=True,
                                             split='train')

    if load_test:
        test_dataset = datasets.MovingMNIST(root=data_dir,
                                            download=True,
                                            split='test')

    return train_dataset, test_dataset


def get_kinetics_dataset(data_dir, arch, load_train=True, load_test=True, batch_size=5):
    """Kinetics-400/600/700 are action recognition video datasets. This dataset consider 
       every video as a collection of video clips of fixed size, specified by frames_per_clip, 
       where the step in frames between each clip is given by step_between_clips.
    """
    train_dataset = test_dataset = None
    if load_train:
        train_dataset = datasets.Kinetics(root=data_dir,
                                          frames_per_clip=5,
                                          num_classes='400',
                                          split='train',
                                          download=True)
        
    if load_test:
        test_dataset = datasets.Kinetics(root=data_dir,
                                         frames_per_clip=5,
                                         num_classes='400',
                                         split='test',
                                         download=True)

    return train_dataset, test_dataset


if __name__ == "__main__":
    from src import project_dir, dataset_dirs
    from src.models import create_model
    import torchtext
    from functools import partial
    from src.utils import SegmentationMeter, ObjectDetectionMeter, TextClassificationMeter, TranslationMeter, VideoProcessingMeter
    from src.dataset_utils import YoloLoss



    model_name = 'fcn_resnet101'
    dataset = 'voc_seg'

    model = create_model(model_name, dataset)
    train, valid, test = load_data(
        dataset, dataset_dirs[dataset], model_name,
        batch_size=1, workers=4, validation_split=0,
        train_size=0.1, valid_size=0.1, test_size=0.1,
        test_only=False, verbose= True
    )
    data, target = next(iter(test))
    out = model(data)
    exit()