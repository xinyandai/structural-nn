import torch
import argparse
from datetime import datetime
from torch.optim import SGD as Optimizer


from dataloaders import *
from models import *
from logger import Logger


NETWORK         = None
COMPRESSOR      = None
DATASET_LOADER  = None
LOGGER          = None
LOSS_FUNC       = nn.CrossEntropyLoss()


network_choices = {
    'resnet18'  : ResNet18,
    'resnet34'  : ResNet34,
    'resnet50'  : ResNet50,
    'resnet101' : ResNet101,
    'resnet152' : ResNet152,
    'vgg11'     : vgg11,
    'vgg13'     : vgg13,
    'vgg16'     : vgg16,
    'vgg19'     : vgg19,
    'dense'     : densenet_cifar,
    'fcn'       : FCN,
    'vqfcn'     : VQFCN,
    'cnn'       : CNN
}

data_loaders = {
    'mnist':    minst,
    'cifar10':  cifar10,
    'cifar100': cifar100,
    'stl10':    stl10,
    'svhn':     svhn,
    'tinyimg':  tinyimgnet
}

classes_choices = {
    'mnist':    10,
    'cifar10':  10,
    'cifar100': 100,
    'stl10':    10,
    'svhn':     10,
    'tinyimg':  200
}


def get_config(args):
    global LOGGER
    global NETWORK
    global DATASET_LOADER

    NETWORK = network_choices[args.network]
    DATASET_LOADER = data_loaders[args.dataset]
    args.num_classes = classes_choices[args.dataset]

    if args.logdir is None:
        assert False, "The logdir is not defined"
    LOGGER = Logger(args.logdir)

    args.no_cuda = args.no_cuda or not torch.cuda.is_available()


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gradient Quantization Samples')
    parser.add_argument('--network', type=str, default='fcn', choices=network_choices.keys())
    parser.add_argument('--dataset', type=str, default='mnist', choices=data_loaders.keys())
    parser.add_argument('--num-classes', type=int, default=10, choices=classes_choices.values())

    parser.add_argument('--logdir', type=str, default=None,
                        help='For Saving the logs')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                        help='weight decay momentum (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                        help='logging training status at each epoch')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--m', type=int, default=1,
                        help='')
    parser.add_argument('--depth', type=int, default=1,
                        help='')
    args = parser.parse_args()
    get_config(args)

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.no_cuda else "cuda")

    train_loader, test_loader = DATASET_LOADER(args)
    model = NETWORK(num_classes=args.num_classes).to(device)
    optimizer = Optimizer(model.parameters(), lr=0.1,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset == 'mnist':
        epochs = [10, 20]
        lrs = [0.01, 0.001]
        args.epochs = 30
    else:
        epochs = [51, 71]
        lrs = [0.01, 0.005]
        args.epochs = 150

    for epoch in range(1, args.epochs + 2):
        for i_epoch, i_lr in zip(epochs, lrs):
            if epoch == i_epoch:
                optimizer = Optimizer(model.parameters(), lr=i_lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)

        origin_train(args, model, device, train_loader, optimizer, epoch, test_loader)

    if args.save_model:
        filename = "saved_{}_{}.pt".format(args.network, datetime.now())
        torch.save(model.state_dict(), filename)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += LOSS_FUNC(output, target).sum().item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def origin_train(args, model, device, train_loader, optimizer, epoch, test_loader):
    model.train()

    iteration = len(train_loader.dataset)//(args.batch_size) + \
        int(len(train_loader.dataset) % (args.batch_size) != 0)
    log_interval = [iteration // args.log_epoch * (i+1) for i in range(args.log_epoch)]

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx + 1 in log_interval:
            print('Train Epoch: {} [{}/{} '
                  '({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            test_accuracy = test(model, device, test_loader)
            info = {'loss': loss.item(), 'accuracy(%)': test_accuracy*100}

            for tag, value in info.items():
                LOGGER.scalar_summary(
                    tag, value, iteration*(epoch-1)+batch_idx)


if __name__ == "__main__":
    main()
