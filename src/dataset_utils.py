import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from torchvision import transforms


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


class YoloLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        anchors=(
                (1.08, 1.19),
                (3.42, 4.41),
                (6.63, 11.38),
                (9.42, 5.11),
                (16.62, 10.52)),
        seen=0,
        coord_prefill=12800,
        threshold=0.6,
        lambda_coord=1.0,
        lambda_noobj=1.0,
        lambda_obj=5.0,
        lambda_cls=1.0):

        super().__init__()

        if not torch.is_tensor(anchors):
            anchors = torch.tensor(anchors, dtype=torch.get_default_dtype())
        else:
            anchors = anchors.data.to(torch.get_default_dtype())

        self.register_buffer("anchors", anchors)

        self.seen = int(seen+.5)
        self.coord_prefill = int(coord_prefill+0.5)

        self.threshold = float(threshold)
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)

        self.mse = torch.nn.MSELoss(reduction='sum')
        self.cel = torch.nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, y):

        nT = y.shape[1]
        nA = self.anchors.shape[0]
        nB, _, nH, nW = x.shape
        nPixels = nH * nW
        nAnchors = nA * nPixels
        y = y.to(dtype=x.dtype, device=x.device)
        x = x.view(nB, nA, -1, nH, nW).permute(0,1,3,4,2)
        nC = x.shape[-1] - 5
        self.seen += nB

        anchors = self.anchors.to(dtype=x.dtype, device=x.device)
        coord_mask = torch.zeros(nB, nA, nH, nW, 1, requires_grad=False, dtype=x.dtype, device=x.device)
        conf_mask = torch.ones(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device) * self.lambda_noobj
        cls_mask = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=torch.bool, device=x.device)
        tcoord = torch.zeros(nB, nA, nH, nW, 4, requires_grad=False, dtype=x.dtype, device=x.device)
        tconf = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device)
        tcls = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device)

        coord = torch.cat([
            x[:, :, :, :, 0:1].sigmoid(),
            x[:, :, :, :, 1:2].sigmoid(),
            x[:, :, :, :, 2:3],
            x[:, :, :, :, 3:4]
        ], -1)

        range_y, range_x = torch.meshgrid(
            torch.arange(nH, dtype=x.dtype, device=x.device),
            torch.arange(nW, dtype=x.dtype, device=x.device)
        )
        anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

        x = torch.cat([
            (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]),
            (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]),
            (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None]),
            (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]),
            x[:, :, :, :, 4:5].sigmoid(),
            x[:, :, :, :, 5:]
        ], -1)
        
        conf = x[..., 4]
        cls = x [..., 5:].reshape(-1, nC)
        x = x[..., :4].detach()

        if self.seen < self.coord_prefill:
            coord_mask.fill_(np.sqrt( .01 / self.lambda_coord))
            tcoord[..., 0].fill_(0.5)
            tcoord[..., 1].fill_(0.5)

        for b in range(nB):
            gt = y[b][(y[b, :, -1] >= 0)[:, None].expand_as(y[b])].view(-1, 6)[:,:4]
            gt[:, ::2] *= nW
            gt[:, 1::2] *= nH
            if gt.numel() == 0:
                continue

            iou_gt_pred = iou(gt, x[b:(b+1)].view(-1, 4))
            mask = (iou_gt_pred > self.threshold).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            #find best anchor for each gt
            iou_gt_anchors = iou_wh(gt[:,2:], anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            #set masks and target values for each gt
            nGT = gt.shape[0]
            gi = gt[:, 0].clamp(0, nW-1).long()
            gj = gt[:, 1].clamp(0, nH-1).long()

            conf_mask[b, best_anchors, gj, gi] = self.lambda_obj
            tconf[b, best_anchors, gj, gi] = iou_gt_pred.view(nGT, nA, nH, nW)[torch.arange(nGT), best_anchors, gj, gi]
            coord_mask[b, best_anchors, gj, gi, :] = (2 - (gt[:, 2] * gt [:, 3]) / nPixels)[..., None]
            tcoord[b, best_anchors, gj, gi, 0] = gt[:, 0] - gi.float()
            tcoord[b, best_anchors, gj, gi, 1] = gt[:, 1] - gj.float()
            tcoord[b, best_anchors, gj, gi, 2] = (gt[:, 2] / anchors[best_anchors, 0]).log()
            tcoord[b, best_anchors, gj, gi, 3] = (gt[:, 3] / anchors[best_anchors, 1]).log()
            cls_mask[b, best_anchors, gj, gi] = 1
            tcls[b, best_anchors, gj, gi] = y[b, torch.arange(nGT), -1]

        coord_mask = coord_mask.sqrt()
        conf_mask = conf_mask.sqrt()
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).expand(nB*nA*nW*nH, nC)
        cls = cls[cls_mask].view(-1, nC)

        loss_coord = self.lambda_coord * self.mse(coord*coord_mask, tcoord*coord_mask) / (2 * nB)
        loss_conf = self.mse(conf*conf_mask, tconf*conf_mask) / (2 * nB)
        loss_cls = self.lambda_cls * self.cel(cls, tcls) / nB

        return loss_coord + loss_conf + loss_cls
        # return loss_coord + loss_conf + loss_cls, [loss_coord.detach().cpu(), loss_conf.detach().cpu(), loss_cls.detach().cpu()]
