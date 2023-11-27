import os
import torch, torchvision
from dataset import MedDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

def fetch_dataloader(dataset, split, batch_size):
    dataset = MedDataset(dataset_name=dataset, split=split, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def fetch_train_dataloader(args):
    dataset = args.dataset
    train_loader = fetch_dataloader(dataset, 'train', args.batch_size)
    return train_loader

def fetch_val_dataloader(args):
    dataset = args.dataset
    val_loader = fetch_dataloader(dataset, 'val', args.batch_size)
    return val_loader

def fetch_test_dataloader(args):
    dataset = args.dataset
    test_loader = fetch_dataloader(dataset, 'test', args.batch_size)
    return test_loader

def evaluate_accuracy(model, dataloader, return_dict = True):
    model.eval()
    preds = []
    labels = []
    for batch in dataloader:
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        _, _, pred = model(x)
        pred = pred.argmax(dim=1)
        preds.append(pred)
        labels.append(y)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    acc_score = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    if return_dict:
        evaluation = classification_report(preds.cpu().numpy(), labels.cpu().numpy(), output_dict = True)
        return acc_score, evaluation
    else:
        return acc_score

def save_prototypes(model, save_dir):
    model.eval()
    prototypes = model.prototypes
    torch.save(prototypes, save_dir + "prototypes.pt")

    if not os.path.exists(save_dir + "prototypes/"):
        os.makedirs(save_dir + "prototypes/")
    
    save_dir = save_dir + "prototypes/"

    for i,prototype in enumerate(prototypes):
        prototype = prototype.detach()
        proto_img = model.decoder(prototype.view(1,64,3,3))
        torchvision.utils.save_image(proto_img, save_dir + f"prototype_{i}.png")



"""BreastMNISTWeights - 
[1, 0.33]

RetinaMNIST - 
[0.2,0.5,0.333,0.333,1]
[0.2,0.8,0.333,0.333,1]

PneumoniaMNIST - 
[1, 0.33]
"""