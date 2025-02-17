from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Train(epochs,train_loader,val_loader,criterion,optmizer,device, regularizer):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    lowest_loss = 0
    
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model  #
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            if regularizer:
                l1 = 0
                for p in net.parameters():
                    l1 = l1 + p.abs().sum()
                lambda_l1 = 0.001
                loss = loss + lambda_l1*l1
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)

        #validate the model#
        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            if regularizer:
                val_l1 = 0
                for p in net.parameters():
                    val_l1 = val_l1 + p.abs().sum()
                lambda_val_l1 = 0.001
                loss = loss + lambda_val_l1*val_l1
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        if validation_loss <= lowest_loss:
            lowest_loss = validation_loss
            torch.save(net.state_dict(), '[MODEL]deep_emotion-{}-{}-{}-{}-{}_best.pt'.format(epochs,batchsize,lr,validation_loss, val_acc))
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))
    
    torch.save(net.state_dict(),'[MODEL]deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str,required= True,
                               help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-hparams', '--hyperparams', type=bool,
                               help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-m', '--channel50', type=bool, help= 'To modify the channel number as 50, set True')
    parser.add_argument('-n', '--stn', type=bool, help= 'To remove stn process, set False')
    parser.add_argument('-e', '--epochs', type= int, help= 'To apply regulization, set True')
    parser.add_argument('-lr', '--learning_rate', type= float, help= 'value of learning rate')
    parser.add_argument('-bs', '--batch_size', type= int, help= 'training/validation batch size')
    parser.add_argument('-l', '--regularizer', type=bool, help='True when adding L1 regulization')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    args = parser.parse_args()

    if args.setup :
        generate_dataset = Generate_data(args.data)
        generate_dataset.split_test()
        generate_dataset.save_images()
        generate_dataset.save_images('finaltest')
        generate_dataset.save_images('val')

    if args.hyperparams:
        epochs = args.epochs
        lr = args.learning_rate
        batchsize = args.batch_size
    else :
        epochs = 100
        lr = 0.005
        batchsize = 128
    
    if args.channel50:
        num_channel = 50
    else:
        num_channel = 10
    
    if args.stn:
        stn = args.stn
    else:
        stn = True
        
    if args.train:
        net = Deep_Emotion(num_channel, stn)
        net.to(device)
        print("Model archticture: ", net)
        traincsv_file = args.data+'/'+'train.csv'
        validationcsv_file = args.data+'/'+'val.csv'
        train_img_dir = args.data+'/'+'train/'
        validation_img_dir = args.data+'/'+'val/'
        
        if args.regularizer:
            regularizer = args.regularizer
        else:
            regularizer = False
    
        transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = 'train', transform = transformation)
        validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
        train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
        val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

        criterion= nn.CrossEntropyLoss()
        optmizer= optim.Adam(net.parameters(),lr= lr)
        Train(epochs, train_loader, val_loader, criterion, optmizer, device, regularizer)
