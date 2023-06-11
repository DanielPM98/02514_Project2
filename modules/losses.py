import torch

def bce_loss(target, pred):
    return torch.mean(pred - target*pred + torch.log(1 + torch.exp(-pred)))

def bce_loss_adjusted(target, pred):
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    return bce_loss(target, pred)

def dice_loss(target, pred):
    return 1 - torch.mean(2 * target * pred + 1) / (torch.mean(target + pred) + 1)

def iou_loss(target, pred):
    intersection = (pred & target).float().sum((1,2))
    union = (pred | target).float().sum((1,2))
    iou = intersection / union
    return 1 - iou.mean()

def focal_loss(y_real, y_pred):
    gamma = 2
    sigma = torch.sigmoid(y_pred)
    p_t = sigma * y_real + (1 - sigma) * (1 - y_real)
    # return - ((1 - sigma)**gamma) * (y_real * torch.log(sigma)) + (1- y_real) * torch.log(1- sigma)
    out = bce_loss(y_real, y_pred) * ((1 - p_t)**gamma)
    # print(out.shape)
    return out.mean()