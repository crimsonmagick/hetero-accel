import torchvision
import torch
from torchvision.models.detection import ssd300_vgg16, retinanet_resnet50_fpn, SSD300_VGG16_Weights, RetinaNet_ResNet50_FPN_Weights
from pycocotools.coco import COCO
from coco_meter import COCOMeter # Correct path for COCO meter to be added
from tqdm import tqdm
import argparse


class CocoDetection(torchvision.datasets.CocoDetection):
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
    
def get_transform(weights):
    weights_transforms = weights.transforms()
    return lambda img, target: (weights_transforms(img), target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOC Segmentation Evaluation")
    parser.add_argument('--device', help = 'cpu or cuda:(GPU number)')
    parser.add_argument('--model_type', default = "SSD300_VGG16", help = "fcn_resnet50 only supported")
    parser.add_argument('--checkpoint_path')

    args = parser.parse_args()
    
    device = args.device


    # Initialize model with checkpoint path

    if args.model_type == 'SSD300_VGG16':
        weights = SSD300_VGG16_Weights.COCO_V1
        model = ssd300_vgg16()
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'],  strict = False)
        
    elif args.model_type == 'RetinaNet_ResNet50':
        weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.retinanet_resnet50_fpn()
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'],  strict = False)


    model = model.to(device)
    model.eval()



    # Step 2: Initialize the inference transforms
    transforms = get_transform(weights)
    root_path = "COCO_DATASET" # path for COCO dataset
    ann_file = f"{root_path}/annotations/instances_val2017.json"

    coco_dataset = CocoDetection(f"{root_path}/val2017",
                                    ann_file = ann_file,
                                    transforms = transforms)


    test_loader = torch.utils.data.DataLoader(coco_dataset, batch_size = 1)


    coco_gt = COCO(ann_file)
    coco_meter = COCOMeter(coco_gt, "bbox")
    print("Inference started")

    for inputs, targets in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        res = {int(targets['image_id']): outputs[0]}
        coco_meter.add(res)
        
    out = coco_meter.value("all")
    print(out)
