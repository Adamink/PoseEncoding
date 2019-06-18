from parser import parser
from model import LSTM
from feeder import Feeder

import torch
import torch.nn as nn
import os
import sys
import numpy as np
import random
import time
from tqdm import tqdm

def load_data():
    print("==> loading train data")
    data_path = os.path.join(args.dataset_dir, 'train_data.npy')
    label_path = os.path.join(args.dataset_dir, 'train_label.pkl')
    valid_frame_path = os.path.join(args.dataset_dir, 'train_num_frame.npy')
    train_loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path, valid_frame_path,
        normalization = args.normalization,
        ftrans = args.ftrans),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    print("==> loading test data")
    data_path = os.path.join(args.dataset_dir, 'test_data.npy')
    label_path = os.path.join(args.dataset_dir, 'test_label.pkl')
    valid_frame_path = os.path.join(args.dataset_dir, 'test_num_frame.npy')
    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path, valid_frame_path,
        normalization = args.normalization,
        ftrans = args.ftrans),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    return (train_loader, test_loader)

def create_model():
    model = LSTM(
        input_size = input_size, 
        num_classes = num_classes, 
        hidden = args.hidden_unit, 
        num_layers = args.num_layers,
        mean_after_fc = args.mean_after_fc,
        mask_empty_frame = args.mask_empty_frame
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    return (model, optimizer)

def load_model():
    print("==> loading existing lstm model")
    model_info = torch.load(model_path)
    model = LSTM(
        input_size = input_size,
        num_classes = model_info['num_classes'],
        hidden = model_info['hidden'],
        num_layers = model_info['num_layers'],
        mean_after_fc = model_info['mean_after_fc'],
        mask_empty_frame = model_info['mask_empty_frame']
    )
    model.cuda()
    model.load_state_dict(model_info['state_dict'])
    best_acc = model_info['best_acc']
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    optimizer.load_state_dict(model_info['optimizer'])
    return (model, optimizer)

def save_model(state):
    torch.save(state,model_path)

def save_acc_loss(train_loss_list, test_loss_list, train_acc_list, test_acc_list):
    loss_file_name = 'loss_' + args.version 
    acc_file_name = 'acc_' + args.version 
    loss_file_path = os.path.join(args.checkpoint_folder, loss_file_name)
    acc_file_path = os.path.join(args.checkpoint_folder, acc_file_name)
    np.save(loss_file_path + '_train.npy', np.asarray(train_loss_list))
    np.save(loss_file_path + '_test.npy', np.asarray(test_loss_list))
    np.save(acc_file_path + '_train.npy', np.asarray(train_acc_list))
    np.save(acc_file_path + '_test.npy', np.asarray(test_acc_list))

def draw_acc_fig():
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    acc_file_name = 'acc_' + args.version 
    acc_file_path = os.path.join(args.checkpoint_folder, acc_file_name)
    train_acc = np.load(acc_file_path + '_train.npy')
    test_acc = np.load(acc_file_path + '_test.npy')
    fig_name = 'acc_' + args.version + '.png'
    fig_path = os.path.join(args.figure_folder, fig_name)
    from visualizer import draw_acc
    draw_acc(train_acc, test_acc, fig_path)
    
def log(message):
    with open(log_path, 'a+') as logfile:
        print(message, file = logfile)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_train_loss = 0.
    correct = 0
    for batch_id, (data, target, frame_num) in enumerate(train_loader):
        data, target, frame_num = \
            data.float().cuda(), target.long().cuda(), frame_num.long().cuda()
        output = model(data, target, frame_num)
        loss = criterion(output, target)
        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.zero_grad()  
        loss.backward()
        total_train_loss += loss.item()
        optimizer.step() 

    train_loss = total_train_loss / len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)
    print('Training: Epoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(epoch,
    total_train_loss, correct, len(train_loader.dataset), 100 * acc))
    return acc, total_train_loss
    
def test(model, test_loader, criterion, epoch, best = False):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target, frame_num in test_loader:
            data, target, frame_num = \
                data.float().cuda(), target.long().cuda(), frame_num.long().cuda()
            output = model(data, target, frame_num)
            test_loss += criterion(output, target)
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    if not best:   
        print('Testing : Epoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    else:
        print("BestPerformance:\nEpoch:{:>3}, Total loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)".format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_acc, test_loss


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, start_epoch = 1):
    print("==> print model properties")
    print(model.__dict__)
    total_loss = 0.
    best_acc = 0.
    best_epoch = 1
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []

    print("==> training model")
    try:
        epoch = start_epoch
        for i in range(args.epochs):
            start = time.time()
            train_acc, train_loss = train(model, train_loader, criterion, optimizer, epoch)
            test_acc, test_loss = test(model, test_loader, criterion, epoch, False)

            elapsed_time = time.time() - start
            s = time.strftime("%M:%S", time.gmtime(elapsed_time))
            print("Time    :    {}".format(s)) 

            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            log("Epoch {:3d}, Train Acc {:.2%}, Test Acc {:.2%}".format(epoch, train_acc, test_acc))

            if test_acc > best_acc:
                best_epoch = epoch
                best_acc = test_acc
                save_model({
                'epoch': epoch,
                'input_size': input_size,
                'num_classes': num_classes,
                'hidden': args.hidden_unit,
                'num_layers': args.num_layers,
                'mean_after_fc': args.mean_after_fc,
                'mask_empty_frame': args.mask_empty_frame,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()})
            
            epoch += 1
    finally:
        # save and draw acc and loss curve
        save_acc_loss(train_loss_list, test_loss_list, train_acc_list, test_acc_list)
        draw_acc_fig()
        # best mode output
        print("==> testing model")
        model_info = torch.load(model_path)
        model.load_state_dict(model_info['state_dict'])
        test_acc, test_loss = test(model, test_loader, criterion, best_epoch, True)
        log("Best Epoch {:3d}, Test Acc {:.2%}".format(best_epoch, test_acc))          

if __name__ == '__main__':
    # parse args
    global args
    args = parser.parse_args()

    # checkpoint path
    global model_path, log_path
    model_name =  'model_' + args.version + '.pt'
    model_path = os.path.join(args.checkpoint_folder, model_name)
    log_name = 'log_' + args.version + '.txt'
    log_path = os.path.join(args.checkpoint_folder, log_name)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(args.gpus)

    # load dataset
    train_loader, test_loader = load_data()
    
    # global info
    global max_frame, num_classes, input_size
    max_frame = train_loader.dataset.data.shape[1]
    input_size = train_loader.dataset.data.shape[-1]
    num_classes = 1 + max(train_loader.dataset.label)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # load or create model
    if args.resume:
        model, optimizer = load_model()
        test(model, test_loader, criterion, 0, True)
    else:
        with open(log_path, "w+") as f:
            pass # clear log
        model, optimizer = create_model()
        train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, 1)

        
    

