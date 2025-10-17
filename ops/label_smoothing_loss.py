import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])

    with torch.no_grad():
        # 空的，没有初始化
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        # 必须要torch.LongTensor()
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)

    return true_dist


if __name__ == '__main__':
    batch_size = 4
    num_class = 63
    smoothing = 0.1

    # loss_fuc = LabelSmoothing(smoothing)
    # loss_fuc = LabelSmoothLoss(smoothing)
    # output = torch.randn([batch_size, num_class])
    # target = torch.randn([batch_size, num_class])

    # print("output: {}".format(output.shape))
    # print("target: {}".format(target.shape))

    # loss = loss_fuc(output, target)
    # print("loss: ", loss)

    true_labels = torch.zeros(2, 5)
    print("true_labels: {}".format(true_labels.shape))

    true_labels[0, 1], true_labels[1, 3] = 1, 1
    print('标签平滑前:\n', true_labels)

    true_dist = smooth_one_hot(true_labels, classes=5, smoothing=0.1)
    print('标签平滑后:\n', true_dist)
    print(true_dist.shape)
