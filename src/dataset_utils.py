import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from torchvision import transforms


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
VOC_CLASSES_MAP = {k: v + 1 for v, k in enumerate(VOC_CLASSES)}
VOC_CLASSES_MAP['background'] = 0
REV_VOC_CLASSES_MAP = {v: k for k, v in VOC_CLASSES_MAP.items()}  # Inverse mapping


def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str: return idx


def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number: return string
    return 'none'


class VOCTransform:
    def __init__(self):
        pass

    def __call__(self,image, target):
        num_bboxes = 10
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width/ img_width, height/img_height)
        new_width, new_height = int(img_width * scale), int( img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = transforms.functional.resize(image, size=(new_height, new_width))
        image = transforms.functional.pad(image, padding = (diff_width//2,
                                                            diff_height//2,
                                                            diff_width//2 + diff_width % 2,
                                                            diff_height//2 + diff_height % 2))

        target = target['annotation']['object']

        target_vectors = []
        for item in target:
            x0 = int(item['bndbox']['xmin'])
            x1 = int(item['bndbox']['xmax'])
            y0 = int(item['bndbox']['ymin'])
            y1 = int(item['bndbox']['ymax'])

            target_vector = [(diff_width/2 + (x0 + x1)/2) / (img_width + diff_width),
                    (diff_height/2 + (y0 + y1)/2) / (img_height + diff_height),
                    (max(x0, x1) - min(x0, x1)) / (img_width + diff_width),
                    (max(y0, y1) - min(y0, y1)) / (img_height + diff_height),
                    1.0,
                    class_to_num(item['name'])]
            

            target_vectors.append(target_vector)
        target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))
        target_vectors = torch.tensor(target_vectors)
        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        return transforms.functional.to_tensor(image), target_vectors


def save_image(image, file_path='image.png'):
    _, ax = plt.subplots()
    ax.imshow(image.numpy()[0,:].transpose(1,2,0))
    plt.savefig('image.png')
    plt.clf()


def save_image_with_bbox(image, bboxes, target_bboxes, file_path='image.png'):
    _, ax = plt.subplots()

    ax.imshow(image.numpy()[0,:].transpose(1,2,0))
    
    
    for i in range(bboxes.shape[1]):

        if bboxes[0,i,-1] > 0:
            cx = bboxes[0,i,0]*320 - bboxes[0,i,2]*320//2
            cy = bboxes[0,i,1]*320 - bboxes[0,i,3]*320//2

            w = bboxes[0,i,2]*320
            h = bboxes[0,i,3]*320

            rect = patches.Rectangle((cx,cy),
                                    w, h, linewidth=2, facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            ax.annotate(num_to_class(int(bboxes[0,i,5])) + " "+  f"{float(bboxes[0,i,4]):.2f}",(cx,cy), color='r')

    for i in range(target_bboxes.shape[1]):
        if target_bboxes[0,i,-1] > 0:
            cx = target_bboxes[0,i,0]*320 - target_bboxes[0,i,2]*320//2
            cy = target_bboxes[0,i,1]*320 - target_bboxes[0,i,3]*320//2

            w = target_bboxes[0,i,2]*320
            h = target_bboxes[0,i,3]*320

            rect = patches.Rectangle((cx,cy),
                                    w, h, linewidth=2, facecolor='none', edgecolor='g')
            ax.add_patch(rect)
            ax.annotate(num_to_class(int(target_bboxes[0,i,5])),(cx,cy), color='g')
    plt.savefig('image.png')
    plt.clf()


def filter_boxes(output_tensor, threshold):
    b, a, h, w, c = output_tensor.shape
    x = output_tensor.contiguous().view(b, a * h * w, c)

    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]
    scores, idx = torch.max(x[:, :, 5:], -1)
    idx = idx.float()
    scores = scores * confidence
    mask = scores > threshold

    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = torch.cat([c[m, :], s[m, None], i[m, None]], -1)
        else:
            detected = torch.zeros((0, 6), dtype=x.dtype, device=x.device)
        filtered.append(detected)
    return filtered


