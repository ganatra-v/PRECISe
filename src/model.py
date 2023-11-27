import torch
import torch.nn as nn
import logging

class network(nn.Module):
  def __init__(self, n_prototypes, num_outputs, in_channels, prototype_dim):
    super(network, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size= 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Dropout2d(p=0.1),
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Dropout2d(p=0.1),
        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels = 32, out_channels = in_channels, kernel_size = 2, stride = 2),
        nn.Sigmoid()
    )

    self.linear = nn.Linear(n_prototypes, num_outputs)
    self.softmax = nn.Softmax(dim=1)

    self.prototypes = nn.Parameter(data = torch.rand(n_prototypes, prototype_dim), requires_grad = True)
    logging.info(f"Initialized prototypes with shape {self.prototypes.shape}")

  def forward(self, x):
    encodings = self.encoder(x)
    reconstructions = self.decoder(encodings)
    flattened = encodings.view(encodings.shape[0], -1)
    distances = torch.cdist(flattened, self.prototypes)
    preds = self.softmax(self.linear(distances))
    return reconstructions, distances, preds