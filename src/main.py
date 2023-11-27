"""
Steps to be followed:
1. Take input from user for dataset, num_prototypes, prototype_dim, 
2. Load corresponding dataset
3. Load model and prototypes
4. Train model
5. Evaluate model
"""

import os
import argparse
import logging
import torch
from constants import Constants

from utils import fetch_train_dataloader, fetch_val_dataloader, fetch_test_dataloader, evaluate_accuracy, save_prototypes
from model import network
import json


parser = argparse.ArgumentParser(description='Prototype-reservation for explainable image classification')
parser.add_argument(
    '--dataset',
    type=str,
    choices= Constants.SUPPORTED_DATASETS,
    required=True,
    help='Dataset to be used for training and testing'
)

parser.add_argument(
    '--prototype_dim',
    type=int,
    help='Dimension of prototypes',
    default=64
)

parser.add_argument(
    '--out_dir',
    type=str,
    help='Output directory',
    required=True
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help='Batch size for training'
)

parser.add_argument(
    '--num_classes',
    type=int,
    required=True,
    help='Number of classes in the dataset'
)

parser.add_argument(
    '--num_channels',
    type=int,
    required=True,
    help="Number of channels in the input image (3 for RGB, 1 for GrayScale)"
)

parser.add_argument(
    '--epochs',
    type=int,
    default=150,
    help='Number of epochs to train the model'
)

def train_loop(args):
    # fetching dataset
    train_loader = fetch_train_dataloader(args)
    val_loader = fetch_val_dataloader(args)

    vae_loss = torch.nn.MSELoss()
    classification_loss = torch.nn.CrossEntropyLoss()
    prototype_loss = torch.nn.MSELoss()
    orientation_loss = torch.nn.MSELoss()

    # loading model
    model = network(
        n_prototypes=args.num_classes * Constants.N_PROTOTYPES_PER_CLASS,
        num_outputs=args.num_classes,
        in_channels=args.num_channels,
        prototype_dim=args.prototype_dim
    )
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay = 1e-4)

    # training model

    best_val = -1


    for epoch in range(args.epochs):
        model.train()
        avg_loss, avg_vloss, avg_closs, avg_ploss, avg_oloss = 0,0,0,0,0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            data = data.cuda()
            target = target.cuda().view(-1)
            reconstructions, distances, preds = model(data.float())
            v_loss = vae_loss(data, reconstructions)
            c_loss = classification_loss(preds, target)

            avg_vloss += v_loss.item()
            avg_closs += c_loss.item()

            loss = v_loss + c_loss
            for i in torch.unique(target):
                subset = distances[target == i][:, i*Constants.N_PROTOTYPES_PER_CLASS : (i+1)*Constants.N_PROTOTYPES_PER_CLASS]
                proto_dist = torch.min(subset, dim=0).values
                orient_dist = torch.min(subset, dim=1).values
                proto_loss = prototype_loss(proto_dist, torch.zeros_like(proto_dist).cuda())
                orient_loss = orientation_loss(orient_dist, torch.zeros_like(orient_dist).cuda())

                loss += Constants.PROTOTYPE_COEFF * (proto_loss + orient_loss)

                avg_ploss += Constants.PROTOTYPE_COEFF * proto_loss.item()/len(torch.unique(target))
                avg_oloss += Constants.PROTOTYPE_COEFF * orient_loss.item()/len(torch.unique(target)) 

            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch: {epoch}, Loss: {avg_loss/ len(train_loader)}, VAE_Loss = {avg_vloss/len(train_loader)}, Prototype_Loss = {avg_ploss/len(torch.unique(target))}, Orientation_Loss = {avg_oloss/len(torch.unique(target))}, Classification_Loss = {avg_closs/len(train_loader)}")
        train_acc_score  = evaluate_accuracy(model, train_loader, return_dict = False)
        acc_score = evaluate_accuracy(model, val_loader, return_dict = False)

        if acc_score > best_val:
            logging.info(f"Found better validation accuracy = {acc_score * 100}%, saving model")
            best_val = acc_score
            torch.save(model.state_dict(), args.out_dir + "/best_model.pt")


        logging.info(f"Epoch: {epoch}, Train Accuracy: {train_acc_score * 100} %, Validation Accuracy: {acc_score * 100} %")

    torch.save(model.state_dict(), args.out_dir + "/model.pt")
    logging.info("Model saved")
    save_prototypes(model, args.out_dir)
    logging.info("Prototypes saved")
    return model, train_loader, val_loader





if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    logging.basicConfig(
        filename=args.out_dir + "/experiments.log",
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filemode='w'
    )

    # model training loop
    model, train_loader, val_loader  =train_loop(args)
    logging.info("Model training complete")

    train_acc, train_report = evaluate_accuracy(model, train_loader)
    val_acc, val_report = evaluate_accuracy(model, val_loader)
    test_loader = fetch_test_dataloader(args)
    test_acc, test_report = evaluate_accuracy(model, test_loader)

    logging.info(f"Train Accuracy: {train_acc * 100} %, Validation Accuracy: {val_acc * 100} %, Test Accuracy: {test_acc * 100} %")

    logging.info(f"Loading best accuracy model")
    model.load_state_dict(torch.load(args.out_dir + "/best_model.pt"))
    logging.info("Model loaded")
    train_acc, train_report_best = evaluate_accuracy(model, train_loader)
    val_acc, val_report_best = evaluate_accuracy(model, val_loader)
    test_acc, test_report_best = evaluate_accuracy(model, test_loader)
    logging.info(f"Train Accuracy: {train_acc * 100} %, Validation Accuracy: {val_acc * 100} %, Test Accuracy: {test_acc * 100} %")

    output = {
        "final_model" : {
            "train" : train_report,
            "val" : val_report,
            "test" : test_report
        },
        "best_model" : {
            "train" : train_report_best,
            "val" : val_report_best,
            "test" : test_report_best
        }
    }
    with open(args.out_dir + "results.json", "w") as f:
        json.dump(output, f)

"""
Class weights - 

BreastMNIST - [1,0.333]
PneumoniaMNIST - [1, 0.333]
RetinaMNIST - [0.2,0.8,0.5,0.5,1]
OCT_MNIST - [0.333, 1, 1, 0.2]
"""
