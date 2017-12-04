import torch
import argparse
from torch import cuda
from torch.autograd import Variable

from pix_dataloader import *
from pix_networks import *
from pix_network_2 import *
from pix_util import *

## don't forget to delete refinement network

# arguments parsed when initiating
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='imagenet', choices=['cifar', 'imagenet', 'celeba', 'mscoco'])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='./models_pix')
    parser.add_argument('--log_path', type=str, default='./logs_pix')
    parser.add_argument('--model', type=str, default='pixcolor100.pkl')
    parser.add_argument('--image_save', type=str, default='./images_pix')
    parser.add_argument('--learning_rate', type=int, default=0.0003)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--resume', type=bool, default=False,
                        help='path to latest checkpoint (default: none)')

    return parser.parse_args()


def main(args):
    dataset = args.data
    gpu = args.gpu
    batch_size = args.batch_size
    model_path = args.model_path
    log_path = args.log_path
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    start_epoch = args.start_epoch

    # make directory for models saved when there is not.
    make_folder(model_path, dataset) # for sampling model
    make_folder(log_path, dataset) # for logpoint model
    make_folder(log_path, dataset +'/ckpt') # for checkpoint model

    # see if gpu is on
    print("Running on gpu : ", gpu)
    cuda.set_device(gpu)

    # set the data-loaders
    train_loader, val_loader, imsize = data_loader(dataset, batch_size)

    # declare class
    RefNet = UNet(imsize)

    # make the class run on gpu
    RefNet.cuda()

    # Loss and Optimizer
    optimizer = torch.optim.Adam(RefNet.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # (int, torch.FloatTensor, torch.LongTensor, torch.FloatTensor, bool) is input haha...

    # optionally resume from a checkpoint
    if args.resume:
        ckpt_path = os.path.join(log_path, dataset, 'ckpt/model.ckpt')
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint")
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            RefNet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            print("=> Meaning that start training from (epoch {})".format(checkpoint['epoch']+1))
        else:
            print("=> Sorry, no checkpoint found at '{}'".format(args.resume))

    # record time
    tell_time = Timer()
    iter = 0
    # Train the Model

    for epoch in range(start_epoch, num_epochs):
        RefNet.train()
        for i, (images, labels) in enumerate(train_loader):
            batch = images.size(0)
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            # make outputs and labels as a matrix for loss calculation
            outputs = images.view(batch, -1)  # 100 x 32*32*3(2048)
            #outputs = RefNet(images)
            labels = labels.contiguous().view(batch, -1)  # 100 x 32*32*3
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.zero_grad()

            if (i + 1) % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.10f, iter_time: %2.2f, aggregate_time: %6.2f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0],
                         (tell_time.toc() - iter), tell_time.toc()))
        torch.save(RefNet.state_dict(), os.path.join(model_path, dataset, 'RefNet%d.pkl' % (epoch + 1)))




        # start evaluation
        print("-------------evaluation start------------")

        RefNet.eval()
        loss_val_all = Variable(torch.zeros(100), volatile=True).cuda()
        for i, (images, _) in enumerate(val_loader):

            # change the picture type from rgb to CIE Lab
            batch = images.size(0)

                # make them all variable + gpu avialable

            images = Variable(images)
            labels = Variable(labels)

            # initialize gradients
            optimizer.zero_grad()

            # make outputs and labels as a matrix for loss calculation
            outputs = images.view(batch, -1)
            outputs = outputs.view(batch, -1)  # 100 x 32*32*3(2048)
            labels = labels.contiguous().view(batch, -1)  # 100 x 32*32*3 igon aniji

            loss_val = criterion(outputs, labels)

            logpoint = {
                'epoch': epoch + 1,
                'args': args,
            }
            checkpoint = {
                'epoch': epoch + 1,
                'args': args,
                'state_dict': RefNet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            loss_val_all[i] = loss_val

            if i == 30:
                print('Epoch [%d/%d], Validation Loss: %.10f'
                      % (epoch + 1, num_epochs, torch.mean(loss_val_all).data[0]))
                torch.save(checkpoint, os.path.join(log_path, dataset, 'ckpt/model.ckpt'))
                break


if __name__ == '__main__':
    args = parse_args()
    main(args)

def make_folder(path, dataset):
    try:
        os.makedirs(os.path.join(path, dataset))
    except OSError:
        pass