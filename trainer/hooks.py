from operator import itemgetter
import os
import datetime

import torch

from tqdm.auto import tqdm

from .utils import AverageMeter

def train_hook_default(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    data_getter=itemgetter("image"),
    target_getter=itemgetter("target"),
    iterator_type=tqdm,
    prefix="",
    stage_progress=False
):
    model = model.train()
    iterator = iterator_type (loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    for i, sample in enumerate(iterator):
        optimizer.zero_grad()
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        predicts = model(inputs)
        loss = loss_fn(predicts, targets)
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.item())
        status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, LR: {4:.5}".format(
            prefix, i, loss_avg.avg, loss_avg.val, optimizer.param_groups[0]["lr"]
        )
        iterator.set_description(status)
    return {"loss": loss_avg.avg}


def test_hook_default(
    model,
    loader,
    loss_fn,
    metric_fn,
    device,
    data_getter=itemgetter("image"),
    target_getter=itemgetter("target"),
    iterator_type=tqdm,
    prefix="",
    stage_progress=False,
    get_key_metric=itemgetter("accuracy")
):
    model = model.eval()
    iterator = iterator_type(loader, disable=not stage_progress, dynamic_ncols=True)
    loss_avg = AverageMeter()
    metric_fn.reset()
    for i, sample in enumerate(iterator):
        inputs = data_getter(sample).to(device)
        targets = target_getter(sample).to(device)
        with torch.no_grad():
            predict = model(inputs)
            loss = loss_fn(predict, targets)
        loss_avg.update(loss.item())
        predict = predict.softmax(dim=1).detach()
        metric_fn.update_value(predict, targets)
        status = "{0}[Test][{1}] Loss_avg: {2:.5}".format(prefix, i, loss_avg.avg)
        if get_key_metric is not None:
            status = status + ", Metric_avg: {0:.5}".format(get_key_metric(metric_fn.get_metric_value()))
        iterator.set_description(status)
    output = {"metric": metric_fn.get_metric_value(), "loss": loss_avg.avg}
    return output

def end_epoch_hook_classification(iterator, epoch, output_train, output_test):

    if hasattr(iterator, "set_description"):
        iterator.set_description(
            "epoch: {0}, test_top1: {1:.5}, train_loss: {2:.5}, test_loss: {3:.5}".format(
                epoch, output_test["metric"]["top1"], output_train["loss"], output_test["loss"]
            )
        )