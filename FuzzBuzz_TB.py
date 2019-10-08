import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from torchbearer.cv_utils import DatasetValidationSplitter
from torchbearer import Trial, VALIDATION_DATA

from models.FuzzBuzzDataset import FuzzBuzzDataset
from models.FuzzBuzzNet import FuzzBuzzModel

from argparse import ArgumentParser


def run(train_batch_size, val_batch_size, epochs, lr, log_interval, input_size=10, hidden_size=100, out_size=4):
    dataset = FuzzBuzzDataset(input_size)
    splitter = DatasetValidationSplitter(len(dataset), 0.1)
    train_set = splitter.get_train_dataset(dataset)
    val_set = splitter.get_val_dataset(dataset)

    train_loader = DataLoader(train_set, pin_memory=True, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, pin_memory=True, batch_size=val_batch_size, shuffle=False, num_workers=2)
    model = FuzzBuzzModel(input_size, hidden_size, out_size)

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()

    trial = Trial(model, optimizer, criterion=loss, metrics=['acc', 'loss']).to(device)
    trial = trial.with_generators(train_generator=train_loader, val_generator=val_loader)
    trial.run(epochs=epochs)

    trial.evaluate(data_key=VALIDATION_DATA)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.log_interval)
