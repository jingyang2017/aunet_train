import torch

def get_f1(y_labels,outputs):
    # f1_ = F1(y_labels,outputs, F1_Thresh=0.5)
    outputs = torch.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    # used in f1
    gd_num = torch.sum(y_ilab, dim=0)
    pr_num = torch.sum(outputs_i, dim=0)

    sum_ones = y_ilab + outputs_i
    pr_rtm = sum_ones // 2

    pr_rt = torch.sum(pr_rtm, dim=0)

    # prevent nan to destroy the f1
    pr_rt = pr_rt.type(torch.float32)
    gd_num = gd_num.type(torch.float32)
    pr_num = pr_num.type(torch.float32)

    zero_scale = torch.zeros_like(torch.min(pr_rt))

    if torch.eq(zero_scale, torch.min(gd_num)):
        gd_num += 1
    if torch.eq(zero_scale, torch.min(pr_num)):
        pr_num += 1
    if torch.eq(zero_scale, torch.min(pr_rt)):
        pr_rt += 0.01

    recall = pr_rt / gd_num
    precision = pr_rt / pr_num
    f1 = 2 * recall * precision / (recall + precision)
    return f1


def get_acc(y_labels,outputs):
    outputs = torch.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    # used in acc
    pr_rtm = torch.eq(outputs_i, y_ilab)
    pr_rt = torch.sum(pr_rtm, dim=0)
    pr_rt = pr_rt.type(torch.float32)
    acc = pr_rt / outputs.shape[0]
    return acc





