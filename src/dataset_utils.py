import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import transforms
from torchinfo import summary


CLASSES = (
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


def iou(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    px, py, pw, ph = bboxes1[...,:4].reshape(-1, 4).split(1, -1)
    lx, ly, lw, lh = bboxes2[...,:4].reshape(-1, 4).split(1, -1)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = torch.tensor(0.0, dtype=px1.dtype, device=px1.device)
    dx = torch.max(torch.min(px2, lx2.T) - torch.max(px1, lx1.T), zero)
    dy = torch.max(torch.min(py2, ly2.T) - torch.max(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])
    
    return ious


def iou_wh(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`
    The bboxes should be defined by their width and height and are centered around (0,0)
    """ 
    w1 = bboxes1[..., 0].view(-1)
    h1 = bboxes1[..., 1].view(-1)
    w2 = bboxes2[..., 0].view(-1)
    h2 = bboxes2[..., 1].view(-1)

    intersections = torch.min(w1[:,None],w2[None,:]) * torch.min(h1[:,None],h2[None,:])
    unions = (w1 * h1)[:,None] + (w2 * h2)[None,:] - intersections
    ious = (intersections / unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious


def nms(filtered_tensor, threshold):
    result = []
    for x in filtered_tensor:
        # Sort coordinates by descending confidence
        scores, order = x[:, 4].sort(0, descending=True)
        x = x[order]
        ious = iou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).long().triu(1).sum(0, keepdim=True).t().expand_as(x) == 0

        result.append(x[keep].view(-1, 6).contiguous())
    return result


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


# dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set="train", download=False, transforms=CustomTransform())
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)



# import utils
# from tinyyolov2 import TinyYoloV2

# net = TinyYoloV2()

# for idx, (input, target) in enumerate(dataloader):
#     #utils.save_image_with_bbox(input, target)
#     out = net(input)
# pass
