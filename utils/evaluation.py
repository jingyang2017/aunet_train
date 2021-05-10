import numpy as np
import torch


def evaluate_metrics(ground_truth,predictions,metrics):
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(ground_truth,predictions)
    return results


def evaluate_flip(net, dataloader_no_flip, dataloader_flip, metrics=None, sel_index=None):
    net.eval()
    # Loop without flip
    for index, data in enumerate(dataloader_no_flip):
        images = data['img'].cuda()
        label = data['label']
        with torch.no_grad():
            pred = net(images)
        pred = pred.data.cpu().float()
        if index:
            preds = torch.cat((preds, pred), dim=0)
            gts = torch.cat((gts, label), dim=0)
        else:
            preds = pred
            gts = label

    if dataloader_flip is not None:
        # Loop with flip
        n_images = 0
        for index, data in enumerate(dataloader_flip):
            images = data['img'].cuda()
            with torch.no_grad():
                pred = net(images)
            pred = pred.data.cpu().float()
            for k in range(0, images.size(0)):
                preds[n_images] = (pred[k] + preds[n_images]) / 2.0
                n_images += 1
    if sel_index is not None:
        preds = preds[:,sel_index]
    results = evaluate_metrics(gts,preds,metrics=metrics)
    net.train()
    return results

def evaluate_flip_fast(net, dataloader_no_flip, dataloader_flip, metrics=None, sel_index=None):
    net.eval()
    index_start=0
    for index, data in enumerate(dataloader_no_flip):
        images = data['img'].cuda()
        label = data['label']
        with torch.no_grad():
            pred = net(images)
        pred = pred.data.cpu().float()

        if index==0:
            preds = torch.zeros(len(dataloader_no_flip.dataset), pred.size(1))
            gts = torch.zeros(len(dataloader_no_flip.dataset), label.size(1))
            index_end = index_start+pred.size(0)
            preds[index_start:index_end,:]=pred
            gts[index_start:index_end,:]=label
            index_start = index_end
        else:
            index_end = index_start+pred.size(0)
            preds[index_start:index_end,:]=pred
            gts[index_start:index_end,:]=label
            index_start = index_end

    assert index_end==len(dataloader_no_flip.dataset)

    if dataloader_flip is not None:
        index_start = 0
        for index, data in enumerate(dataloader_flip):
            images = data['img'].cuda()
            with torch.no_grad():
                pred = net(images)
            pred = pred.data.cpu().float()
            if index==0:
                preds_flip = torch.zeros(len(dataloader_no_flip.dataset), pred.size(1))
                index_end = index_start + pred.size(0)
                preds_flip[index_start:index_end, :] = pred
                index_start = index_end
            else:
                index_end = index_start + pred.size(0)
                preds_flip[index_start:index_end, :] = pred
                index_start = index_end

        assert index_end == len(dataloader_no_flip.dataset)
        preds = (preds+preds_flip)/2
    if sel_index is not None:
        preds = preds[:,sel_index]
    results = evaluate_metrics(gts,preds,metrics=metrics)
    net.train()
    return results
