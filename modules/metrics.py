import torch


class Metrics:
    def __init__(self, eps=1e-5, activation=None, mode='mean'):
        self.eps = eps
        self.activation = activation
        self.mode = mode
        self.metrics = []


    def __call__(self):
        return {
            'accuracy': 0,
            'dice_score': 0,
            'specifity': 0,
            'sensitivity': 0,
            'iou': 0
        }

    def add(self, target, pred):
        if self.activation is not None:
            pred = self.activation(pred)
        
        self.metrics.append(self._calc_metrics(target, pred))

    def mean(self):
        acc_metrics = {
            'accuracy': 0,
            'dice_score': 0,
            # 'precision': precision,
            'specifity': 0,
            'sensitivity': 0,
            'iou': 0
        }
        for metric_ in self.metrics:
            for key, value in metric_.items():
                acc_metrics[key] += value

        for key, value in acc_metrics.items():
            acc_metrics[key] /= len(self.metrics)

        return acc_metrics
    
    def _calc_metrics(self, target, pred):
        pred = (pred > 0).float()
        dice = self._dice_coeff(target, pred)
        pred = pred.view(-1, )
        target = target.view(-1, )

        tp = torch.sum(pred * target)  # TP
        fp = torch.sum(pred * (1 - target))  # FP
        fn = torch.sum((1 - pred) * target)  # FN
        tn = torch.sum((1 - pred) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        # dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        # precision = (tp + self.eps) / (tp + fp + self.eps)
        specifity = (tn + self.eps) / (tn + fp + self.eps)
        sensitivity = (tp + self.eps) / (tp + fn + self.eps)
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)

        output = {
            'accuracy': pixel_acc,
            'dice_score': dice,
            # 'precision': precision,
            'specifity': specifity,
            'sensitivity': sensitivity,
            'iou': iou
        }

        return output
    
    def _dice_coeff(self, target, pred):
        target_f = target.flatten()
        pred_f = pred.flatten()

        intersection = (target_f * pred_f).sum()
        return (2. * intersection + self.eps) / ((target_f + pred_f).sum() + self.eps)





def accuracy(target, pred):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    return correct / target.numel()


def sensitivity(target, pred):
    pred = torch.argmax(pred, dim=1)
    num_classes = len(torch.unique(target))

    conf_matrix = torch.zeros(num_classes, num_classes)

    for t, p in zip(target.view(-1), pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    # Compute the true positives, true negatives, false positives and false negatives
    tp = conf_matrix[1][1]
    fn = conf_matrix[1][0]

    sensitivity = tp / (tp + fn)
    
    return sensitivity


def specifity(target, pred):
    pred = torch.argmax(pred, dim=1)
    num_classes = len(torch.unique(target))

    conf_matrix = torch.zeros(num_classes, num_classes)

    for t, p in zip(target.view(-1), pred.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    # Compute the true positives, true negatives, false positives and false negatives
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]

    specifity = tn / (tn + fp)

    return specifity