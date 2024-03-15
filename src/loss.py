import torch
import numpy as np
from src.utils import iou, iou_wh


__all__ = ['DummyLoss', 'SegLoss', 'YoloLoss']



class DummyLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def to(self, device):
        pass

    def forward(self, x, y):
        return torch.tensor(0.0)


class SegLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.NLLLoss()#ignore_index=-1)

    def to(self, device):
        self._loss.to(device)

    def forward(self, x, y):
        return torch.tensor(0.0)
        # TODO: Figure out why this throws errors
        # return self._loss(x, y)


# NOTE: Alternative to YoloLoss: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py


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
        lambda_cls=1.0
    ):

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
