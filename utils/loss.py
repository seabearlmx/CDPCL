import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HardPrototypeContrastiveLoss(nn.Module):
    def __init__(self):
        super(HardPrototypeContrastiveLoss, self).__init__()

    def forward(self, Proto1, Proto2, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )
        Returns:
        """
        assert not Proto1.requires_grad
        assert not Proto2.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != 255)
        labels = labels[mask]
        feat = feat[mask]

        S = loss_calc_cosin2(Proto1, Proto2)  # 19*19
        I = torch.eye(19).cuda()
        hard_weight = torch.abs(I-S)

        feat = F.normalize(feat, p=2, dim=1)
        Proto1 = F.normalize(Proto1, p=2, dim=1)

        logits = feat.mm(Proto1.permute(1, 0).contiguous())

        hard_weighted_logits = logits.mm(hard_weight)

        hard_weighted_logits = hard_weighted_logits / 0.8

        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(hard_weighted_logits, labels)

        return loss


class PrototypeContrastiveLoss(nn.Module):
    def __init__(self):
        super(PrototypeContrastiveLoss, self).__init__()

    def forward(self, Proto, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )
        Returns:
        """
        assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        mask = (labels != 255)
        labels = labels[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)

        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / 0.8

        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(logits, labels)

        return loss


def loss_calc_cosin(pred1, pred2):
    # n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.abs(1 - (torch.matmul(pred1, pred2) / (torch.norm(pred1) * torch.norm(pred2))))
    return output


def loss_calc_cosin2(pred1, pred2):
    # pred1 = pred1.view(-1).cuda()
    # pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = (torch.matmul(pred1, pred2.permute(1, 0).contiguous()) / (torch.norm(pred1) * torch.norm(pred2)))
    return output


def loss_calc_dist(pred1, pred2):
    n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.sum(torch.abs(pred1 - pred2)) / (h * w * c)
    return output


def inner_class_distance(input_vec):
    B, C, H, W = input_vec.shape
    HW = H * W
    input_vec = F.normalize(input_vec, p=2, dim=1)
    input_vec = input_vec.contiguous().view(B, -1, C)
    # print(input_vec)
    x = input_vec - torch.mean(input_vec, axis=0, keepdim=True)
    cov_matrix = torch.bmm(x.transpose(1, 2), x).div(HW - 1)
    # cov_matrix = torch.matmul(x.T, x) / (B * C * HW - 1)
    # print(cov_matrix)
    # print(torch.diag(cov_matrix).shape)
    # print(torch.sum(torch.diag(cov_matrix)).shape)
    diag = 0
    for i in range(B):
        diag += torch.sum(torch.diag(cov_matrix[i]))
    tr_sum = diag / B
    inner_dist = tr_sum * 2
    return inner_dist


def proto_inner_distance(input_vec):
    input_vec = input_vec.unsqueeze(0)
    C, HW = input_vec.shape
    input_vec = F.normalize(input_vec, p=2, dim=0)
    input_vec = input_vec.contiguous().view(-1, C)
    # print(input_vec)
    x = input_vec - torch.mean(input_vec, axis=0, keepdim=True)
    cov_matrix = torch.matmul(x.transpose(0, 1), x).div(HW - 1)
    # cov_matrix = torch.matmul(x.T, x) / (B * C * HW - 1)
    # print(cov_matrix)
    # print(torch.diag(cov_matrix).shape)
    # print(torch.sum(torch.diag(cov_matrix)).shape)
    # diag = 0
    # for i in range(B):
    #     diag += torch.sum(torch.diag(cov_matrix[i]))
    diag = torch.sum(torch.diag(cov_matrix))
    tr_sum = diag
    inner_dist = tr_sum * 2
    return inner_dist


FG_LABEL = [5,6,7,11,12,13,14,15,16,17,18]
class InsCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(InsCrossEntropyLoss, self).__init__()
        self.id_to_trainid = {0: 5, 1: 6, 2: 7, 3: 11, 4: 12, 5: 13,
                              6: 14, 7: 15, 8: 16, 9: 17, 10: 18}

    def forward(self, predict, labels):
        label_copy = 255 * torch.ones(labels.shape)
        for k, v in self.id_to_trainid.items():
            label_copy[labels == v] = k
        target = label_copy.long().cuda()
        n, c, h, w = predict.size()
        # remove IGNORE_LABEL pixels
        mask = (target >= 0) * (target != 255)
        target = target[mask]

        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, size_average=True)

        return loss