import os
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from torch.optim.lr_scheduler import *
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
from dataloader import Dataset, filter_none
from model import build_model
import pickle
import h5py as h5
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import warnings
import parameters as params
warnings.filterwarnings("ignore")
import json
import random


def train_epoch(cfg, run_id, epoch, data_loader, model, num_classes, optimizer, criterion, writer, use_cuda, args, weights=None, accumulation_steps=1):
    print('train at epoch {}'.format(epoch), flush=True)

    losses = []
    loss1, loss2 = [], []

    model.train()

    for i, (clips, targets) in enumerate(data_loader):
        assert len(clips) == len(targets)

        if use_cuda:
            clips = Variable(clips.type(torch.FloatTensor)).cuda()
            targets = Variable(targets.type(torch.FloatTensor)).cuda()
        else:
            clips = Variable(clips.type(torch.FloatTensor))
            targets = Variable(targets.type(torch.FloatTensor))

        optimizer.zero_grad()

        out = model(clips)

        loss = 0

        for key in out:
            if 'output' in key:
                outputs = out[key]
                outputs = outputs.reshape(-1, num_classes)
                targets = targets.reshape(-1, num_classes)
                _loss = criterion(outputs, targets)
                if weights is None:
                    loss +=  _loss
                else:
                    loss += weights[key][epoch] * _loss
                if key == 'init_output':
                    loss1.append(_loss.item())
                if key == 'final_output':
                    loss2.append(_loss.item())

        loss = loss / accumulation_steps

        loss.backward()

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        del loss, outputs, clips, targets

    print('Training Epoch: %d, Loss: %.4f, Initial: %.4f, Final: %.4f' % (epoch, np.mean(losses), np.mean(loss1), np.mean(loss2)), flush=True)

    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('Training Loss Initial', np.mean(loss1), epoch)
    writer.add_scalar('Training Loss Final', np.mean(loss2), epoch)

    if args.swa_start > 0 and epoch > args.swa_start:
        optimizer.update_swa()
        if epoch % args.swa_update_interval == 0:
            optimizer.swap_swa_sgd()
    return model


def build_labels(video_id, annotations_file, num_features, num_classes, add_background=False):
    annotations = json.load(open(annotations_file, 'r')) 
    labels = np.zeros((num_features, num_classes), np.float32)
    fps = num_features/annotations[video_id]['duration']
    for annotation in annotations[video_id]['actions']:
        for fr in range(0, num_features, 1):
            if fr/fps >= annotation[1] and fr/fps <= annotation[2]:
                labels[fr, annotation[0] - 1] = 1    # will make the first class to be the last for datasets other than Multi-Thumos #
    if add_background == True:
        new_labels = np.zeros((num_features, num_classes + 1))
        for i, label in enumerate(labels):
            new_labels[i,0:-1] = label
            if np.max(label) == 0:
                new_labels[i,-1] = 1
        return new_labels
    return labels


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def val_epoch(cfg, epoch, model, writer, use_cuda, args):
    print('validation at epoch {}'.format(epoch))

    num_gpus = len(args.gpu.split(','))
  
    model.eval()

    video_list = [line.rstrip().replace('.txt', '') for line in open(cfg.test_list, 'r').readlines()]

    if args.input_type == 'rgb':
        assert os.path.exists(cfg.rgb_test_file)
        data = h5.File(cfg.rgb_test_file, 'r')
    elif args.input_type == 'flow':
        assert os.path.exists(cfg.flow_test_file)
        data = h5.File(cfg.flow_test_file, 'r')
    elif args.input_type == 'combined':
        assert os.path.exists(cfg.combined_test_file)
        data = h5.File(cfg.combined_test_file, 'r')

    initial_predictions, predictions, ground_truth = [], [], []
    for i, video in tqdm(enumerate(video_list)):
        features = data[video]
        labels = build_labels(video, cfg.annotations_file, len(features), cfg.num_classes, args.add_background)

        if args.add_background:
            num_classes = cfg.num_classes + 1
        else:
            num_classes = cfg.num_classes

        features = np.array(features)
        labels = np.array(labels)
        features = Variable(torch.from_numpy(features).type(torch.FloatTensor))
        labels =  Variable(torch.from_numpy(labels).type(torch.FloatTensor))
        assert len(features) == len(labels)

        with torch.no_grad():
            if args.num_clips > 0:
                eval_mode = args.eval_mode
                if len(features) < args.num_clips:
                    eval_mode = 'pad'
                if eval_mode == 'truncate':
                    features = features[0:len(features) - (len(features) % args.num_clips)]
                    labels = labels[0:len(labels) - (len(labels) % args.num_clips)]
                    features = torch.stack([features[i:i + args.num_clips] for i in range(0, len(features), args.num_clips)])
                    labels = torch.stack([labels[i:i + args.num_clips] for i in range(0, len(labels), args.num_clips)])
                elif eval_mode == 'pad':
                    features_to_append = torch.zeros(args.num_clips - len(features) % args.num_clips, features.shape[1])
                    labels_to_append = torch.zeros(args.num_clips - len(labels) % args.num_clips, labels.shape[1])
                    features = torch.cat((features, features_to_append), 0)
                    labels = torch.cat((labels, labels_to_append), 0)
                    features = torch.stack([features[i:i + args.num_clips] for i in range(0, len(features), args.num_clips)])
                    labels = torch.stack([labels[i:i + args.num_clips] for i in range(0, len(labels), args.num_clips)]) 
                elif eval_mode == 'slide':
                    slide_rate = 16
                    features_to_append = torch.zeros(slide_rate - len(features) % slide_rate, features.shape[-1])
                    labels_to_append = torch.zeros(slide_rate - len(labels) % slide_rate, labels.shape[-1])
                    features = torch.cat((features, features_to_append), 0)
                    labels = torch.cat((labels, labels_to_append), 0)
                    features = torch.stack([features[i:i + args.num_clips] for i in range(0, len(features) - args.num_clips + 1, slide_rate)])
                    labels = torch.stack([labels[i:i + args.num_clips] for i in range(0, len(labels) - args.num_clips + 1, slide_rate)])
                assert len(features) > 0
            else:
                features = torch.unsqueeze(features, 0)
                labels = torch.unsqueeze(labels, 0)

            features = features.cuda()
            labels = labels.cuda()

            out = model(features)
            outputs = out['final_output']
            initial = out['init_output']

            outputs = nn.Softmax(dim=-1)(outputs)
            initial = nn.Softmax(dim=-1)(initial)

            outputs = outputs.reshape(-1, num_classes)
            initial = initial.reshape(-1, num_classes)
            labels = labels.reshape(-1, num_classes)
            outputs = outputs.cpu().data.numpy()
            initial = initial.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
      
        assert len(outputs) == len(labels)
        predictions.extend(outputs)
        ground_truth.extend(labels)
        initial_predictions.extend(initial)

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    initial_predictions = np.array(initial_predictions)

    avg_precision_score = average_precision_score(ground_truth, predictions, average=None)
    initial_avg_precision_score = average_precision_score(ground_truth, initial_predictions, average=None)
    predictions = (np.array(predictions) > args.f1_threshold).astype(int)
    ground_truth = (np.array(ground_truth) > args.f1_threshold).astype(int)
    results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions), average=None)
    f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]

    if args.add_background:
        avg_precision_score = avg_precision_score[:-1]
        initial_avg_precision_score = initial_avg_precision_score[:-1]
        f1_scores = f1_score[:-1]
    print('Validation Epoch: %d, F1-Score: %s' % (epoch, str(f1_scores)), flush=True)
    print('Validation Epoch: %d, Average Precision: %s' % (epoch, str(avg_precision_score)), flush=True)
    print('Validation Epoch: %d, F1-Score: %4f, mAP: %4f, initial mAP: %4f' 
                                                        % (epoch, np.nanmean(f1_scores), np.nanmean(avg_precision_score), np.nanmean(initial_avg_precision_score)), flush=True)

    writer.add_scalar('Validation F1 Score', np.nanmean(f1_scores), epoch)
    writer.add_scalar('Validation Precision', np.nanmean(precision), epoch)
    writer.add_scalar('Validation Recall', np.nanmean(recall), epoch)
    writer.add_scalar('Validation AP', np.nanmean(avg_precision_score), epoch)
    return np.nanmean(avg_precision_score)


def train_model(cfg, run_id, save_dir, use_cuda, args, writer):
    shuffle = True
    print("Run ID : " + args.run_id)
   
    print("Parameters used : ")
    print("batch_size: " + str(args.batch_size))
    print("lr: " + str(args.learning_rate))
    print("loss weights: " + str(params.weights))

    if args.random_skip:
        skip = [x for x in range(0, 4)]
    else:
        skip = [args.skip]
    train_data_gen = Dataset(cfg, args.input_type, 'training', 1.0, args.num_clips, skip, add_background=args.add_background)
    train_dataloader = DataLoader(train_data_gen, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, 
                                  collate_fn=lambda b:filter_none(b, args.num_clips, args.varied_length))

    print("Number of training samples : " + str(len(train_data_gen)))
    steps_per_epoch = len(train_data_gen) / args.batch_size
    print("Steps per epoch: " + str(steps_per_epoch))

    if args.add_background:
        num_classes = cfg.num_classes + 1
    else:
        num_classes = cfg.num_classes

    assert args.num_clips > 1
    model = build_model(args.model_version, args.num_clips, num_classes, args.feature_dim, args.hidden_dim, args.num_layers)

    num_gpus = len(args.gpu.split(','))
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    if use_cuda:
        model.cuda()

    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.5)

    if args.swa_start > 0:
        optimizer = SWA(optimizer)

    criterion = BCEWithLogitsLoss()

    max_fmap_score, fmap_score = 0, 0
    # loop for each epoch
    for epoch in range(args.num_epochs):
        model = train_epoch(cfg, run_id, epoch, train_dataloader, model, num_classes, optimizer, criterion, writer, use_cuda, args, weights=None, accumulation_steps=args.steps)
        if args.dataset in ['charades']:
            validation_interval = 10
            if epoch > 20:
                validation_interval = 5
        else:
            validation_interval = 50
            if epoch > 1000: 
                validation_interval = 10
        if epoch % validation_interval == 0:
            fmap_score = val_epoch(cfg, epoch, model, writer, use_cuda, args)
         
        if fmap_score > max_fmap_score:
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
            save_file_path = os.path.join(save_dir, 'model_{}_{:.4f}.pth'.format(epoch, fmap_score))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            max_fmap_score = fmap_score
