from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.backends import cudnn
import matplotlib.pyplot as plt
import net
import os
import ResNet
import glob
import numpy as np
def train(epoch, model, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * data.size()[0], len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data))

            # save snapshot
            print('Snapshot save at epoch: %d, batch: %d' % \
                  (epoch, batch_idx + 1))

            snapshot = {'epoch': epoch, \
                        'state_dict': model.state_dict(), \
                        'optimizer': optimizer.state_dict()}
            torch.save(snapshot, "./snapshot"+'/snapshot_' + str(epoch) + '_' + str(batch_idx + 1))

        if (batch_idx + 1) == len(train_loader):
            print('Training of epoch {} finished, snapshot saved'.format(epoch))
            snapshot = {'epoch': epoch, \
                        'state_dict': model.state_dict(), \
                        'optimizer': optimizer.state_dict()}
            torch.save(snapshot, "./snapshot"+ '/snapshot_' + str(epoch) + '_' + str(batch_idx + 1))
    return loss.data

def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss,np.float64(correct) / np.float(len(test_loader.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', \
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', \
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', \
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', \
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', \
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, \
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', \
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N', \
                        help='how many batches to wait before logging training status')
    parser.add_argument('--snapshot', type=str, default='./snapshot', metavar='PATH', \
                        help='snapshot location')
    parser.add_argument('--resume', type=str, metavar='PATH', default='./snapshot/snapshot_4_938', \
                        help='path to latest snapshot (default: none)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print('Using CUDA:' + str(args.cuda))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = ResNet.MnistResNet()
    if args.cuda:
        model.cuda()
    # cudnn.enabled = False
    cudnn.benchmark = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transforms.Compose(
                                       [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST('./data', train=False,
                                  transform=transforms.Compose(
                                      [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if not os.path.isdir(args.snapshot):
        os.mkdir(args.snapshot)
    # else:
    # 	files = glob.glob(args.snapshot+'/*')
    # 	for f in files:
    # 		os.remove(f)

    start_epoch = 1

    ep=np.arange(args.epochs)+1
    if args.epochs < start_epoch:
        print("Epoch number is less than the one in snapshot, training abort")
    else:
        test_result=[]
        train_result=[]
        acc_result=[]
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        for epoch in range(1, args.epochs + 1):
            if epoch >= start_epoch:
                trainloss=train(epoch, model,  args.log_interval)
                loss,acc=test(epoch, model)
                train_result.append(trainloss)
                test_result.append(loss)
                acc_result.append(acc)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        plt.plot(ep,test_result)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('TestSet')
        plt.savefig('TestSet.png')
        plt.close()

        plt.plot(ep,train_result)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('TrainSet')
        plt.savefig('TrainSet.png')
        plt.close()

        plt.plot(ep,acc_result)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy')
        plt.savefig('accuracy.png')
        plt.close()