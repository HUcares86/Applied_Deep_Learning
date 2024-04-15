import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from metric import mapk


def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, cfg):
    torch.set_printoptions(precision=3)
    correct_all = torch.tensor([], device='cuda')
    loss_values = []
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        iters = len(train_loader)
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs, labels = data.to(cfg.device), target.to(cfg.device)

                outputs = model(inputs)
                if cfg.challenge == 'seen':
                    outputs = outputs + inputs[:, cfg.in_features - cfg.course_num:] * -999
                loss = criterion(outputs, labels)

                outputs_pred = torch.argmax(outputs, axis=1)
                gt = torch.argmax(labels, axis=1)

                correct_all = torch.cat((correct_all, (outputs_pred == gt)), 0)
                accuracy = correct_all.sum().item() / correct_all.shape[0]

                optimizer.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if cfg.scheduler == 'cos':
                    scheduler.step(epoch + i / iters)

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)
                epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        loss_values.append(epoch_loss)
        accuracy = valid_model(model, valid_loader, criterion, cfg)
        if cfg.scheduler == 'exp':
            scheduler.step()

        if accuracy > cfg.benchmark:
            cfg.benchmark = accuracy
            print('current accuracy', accuracy, 'beat benchmark, saving model')
            torch.save(model, cfg.model_path)
    print('\nBest Accuracy: ', cfg.benchmark)
    return model


@torch.no_grad()
def valid_model(model, valid_loader, criterion, cfg):
    torch.set_printoptions(precision=3)
    correct_all = torch.tensor([], device='cuda')
    valid_loss = []
    all_accuracy = []

    model.eval()
    with tqdm(valid_loader, unit="batch") as tepoch:
        for data, target, avoid_courses in tepoch:
            tepoch.set_description(f"Validation")

            inputs, labels = data.to(cfg.device), target.to(cfg.device)
            avoid_courses = avoid_courses.to(cfg.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())
            outputs = outputs + avoid_courses * -999

            top50_val, top50_idx = torch.topk(outputs, 50, dim=1)
            top50_idx = top50_idx.cpu().tolist()
            top_idx = []
            for i in range(inputs.shape[0]):
                top_idx.append([x for x, y in zip(top50_idx[i], top50_val[i]) if y > cfg.select_threshold])

            gt = []
            for label in labels:
                gt.append((label == 1).nonzero().flatten().cpu().tolist())

            # print(top50_idx[:, :10])
            # print(top50_idx.shape)
            # print(gt)
            # print(len(gt))
            # exit()

            accuracy = mapk(gt, top_idx, k=50)  # actual, predicted, k
            all_accuracy.append(accuracy)

            # outputs_pred = torch.argmax(outputs, dim=1)
            # gt = torch.argmax(labels, axis=1)

            # correct_all = torch.cat((correct_all, (outputs_pred == gt)), 0)
            # accuracy = correct_all.sum().item() / correct_all.shape[0]

            tepoch.set_postfix(accuracy=sum(all_accuracy)/len(all_accuracy), loss=sum(valid_loss)/len(valid_loss))

    print('total validation accuracy:', sum(all_accuracy)/len(all_accuracy))

    return sum(all_accuracy)/len(all_accuracy)
