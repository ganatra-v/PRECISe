import os
import torch
import json
from model import network
from utils import fetch_test_dataloader, evaluate_accuracy, save_prototypes
from constants import Constants

IN_DIR = "../models/"
OUT_DIR = "../output/"

class args:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

def evaluate_models(in_dir, out_dir):
    models = os.listdir(in_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for model_ in models:
        dataset = model_.split(".pt")[0]
        print(dataset)
        if not os.path.exists(out_dir + dataset):
            os.makedirs(out_dir + dataset)

        args_ = args(dataset, 64)

        test_loader = fetch_test_dataloader(args_)

        if dataset == Constants.BREAST_MNIST:
            model = network(8,2,1,576)
        elif dataset == Constants.PNEUMONIA_MNIST:
            model = network(4,2,1,576)
        elif dataset == Constants.RETINA_MNIST:
            model = network(20,5,3,576)
        elif dataset == Constants.OCT_MNIST:
            model = network(16,4,1,576)
        model.load_state_dict(torch.load(IN_DIR + model_))
        model = model.cuda()
        model.eval()
        test_acc, test_report = evaluate_accuracy(model, test_loader)
        with open(OUT_DIR + dataset + "/results.json", "w") as f:
            json.dump(test_report, f)
        save_prototypes(model, OUT_DIR +  dataset + "/")

evaluate_models(IN_DIR, OUT_DIR)