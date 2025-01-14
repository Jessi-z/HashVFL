import logging
import math
import os
import torch
import numpy as np
import random

from utils.model import Wide_Deep_Model, Server


def setupSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setupLogger(filename, record=True):
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    # setup stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # setup file handler
    if record:
        fh = logging.FileHandler("./logs/{}".format(filename), encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logging.info('Start print log......')
    return logger


def generateFeatureRatios(num_party):
    feature_ratios = [round(1 / num_party, 2) for _ in range(num_party - 1)]
    feature_ratios.append(round(1 - round(1 / num_party, 2) * (num_party - 1), 2))
    return feature_ratios


def feature_distribute(size, feature_ratios):
    num_features = []
    total = 0
    for i in range(len(feature_ratios) - 1):
        num_feature = round(size * feature_ratios[i])
        num_features.append(num_feature)
        total += num_feature
    num_features.append(size - total)
    return num_features


def transFromCode(target):
    value = 0
    for index, i in enumerate(target):
        value += i * math.pow(2, index)
    return value


def generateOrthogonalTargets(num_classes, encode_length):
    if math.pow(2, encode_length) < num_classes:
        logging.info('Too small encode length for orthogonal target generation!')
        exit()
    targets = []
    values = []
    for i in range(num_classes):
        while True:
            target = 2 * np.random.binomial(1, 0.5, encode_length) - 1
            value = transFromCode(target)
            if value in values:
                continue
            else:
                targets.append(target)
                values.append(value)
                break
    return torch.FloatTensor(np.array(targets))


def loadOrthogonalTargets(filename):
    return torch.load(filename)


def prepareModels(dataset_name, num_features, encode_length, defense, epsilon, num_party, num_classes, num_layers,
                  device):
    workers = []
    if dataset_name in ['criteo']:
        for num_feature in num_features:
            workers.append(
                Wide_Deep_Model(in_features=num_feature, out_features=encode_length, defense=defense, epsilon=epsilon,
                    device=device, num_layers=num_layers).to(device))
    else:
        logging.info('Not supported datatype!')
        exit()
    server = Server(num_party=num_party, in_features=encode_length, num_classes=num_classes, num_layers=1).to(device)
    return workers, server


def loadModels(root, dataset_name, num_features, encode_length, defense, epsilon, num_party, num_classes, num_layers,
               device):
    workers = []
    if dataset_name in ['criteo']:
        for i, num_feature in enumerate(num_features):
            worker = Wide_Deep_Model(in_features=num_feature, out_features=encode_length, defense=defense, epsilon=epsilon,
                         device=device, num_layers=num_layers).to(device)
            path = os.path.join(root, 'worker_{}.pt'.format(i))
            worker.load_state_dict(torch.load(path))
            workers.append(worker)
    else:
        logging.info('Not supported datatype!')
        exit()
    server = Server(num_party=num_party, in_features=encode_length, num_classes=num_classes, num_layers=1).to(
        device)
    path = os.path.join(root, 'server.pt')
    server.load_state_dict(torch.load(path))
    return workers, server


def loadBottomModels(dataset_name, num_features, encode_length, epsilon, defense, device, num_layers=1):
    workers = []
    if dataset_name in ['criteo']:
        for num_feature in num_features:
            workers.append(
                Wide_Deep_Model(in_features=num_feature, out_features=encode_length, defense=defense, epsilon=epsilon,
                    device=device, num_layers=num_layers).to(device))
    else:
        logging.info('Not supported datatype!')
        exit()
    return workers
