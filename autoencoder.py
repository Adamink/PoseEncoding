from feeder import Feeder
from main import save_model, log

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_size, norm = True):
        super(Autoencoder, self).__init__()
        if not norm:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 40),#nn.ReLU(True),
                nn.Linear(40, 30),#nn.ReLU(True), 
                nn.Linear(30, 20), nn.ReLU(True))
            self.decoder = nn.Sequential(
                nn.Linear(20, 30),#nn.Tanh(True),
                nn.Linear(30, 40),#nn.Tanh(True),
                nn.Linear(40, input_size),nn.Tanh())
        else:
            self.encoder = nn.Sequential(
                weight_norm(nn.Linear(input_size, 40)),#nn.ReLU(True),
                weight_norm(nn.Linear(40, 30)),#nn.ReLU(True), 
                weight_norm(nn.Linear(30, 20)), nn.ReLU(True))
            self.decoder = nn.Sequential(
                weight_norm(nn.Linear(20, 30)),#nn.Tanh(True),
                weight_norm(nn.Linear(30, 40)),#nn.Tanh(True),
                weight_norm(nn.Linear(40, input_size)),nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_hidden(self, x):
        x = self.encoder(x)
        return x

def load_train_data():
    print("==> loading train data")
    data_path = os.path.join(args.dataset_dir, 'all_data.npy')
    label_path = os.path.join(args.dataset_dir, 'all_label.pkl')
    valid_frame_path = os.path.join(args.dataset_dir, 'all_num_frame.npy')
    train_loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path, valid_frame_path,
        normalization = args.normalization,
        ftrans = args.ftrans,
        reshape=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    return train_loader

def load_input_data():
    print("==> loading train data")
    data_path = os.path.join(args.dataset_dir, 'train_data.npy')
    label_path = os.path.join(args.dataset_dir, 'train_label.pkl')
    valid_frame_path = os.path.join(args.dataset_dir, 'train_num_frame.npy')
    train_feeder = Feeder(data_path, label_path, valid_frame_path,
        normalization = args.normalization,
        ftrans = args.ftrans,
        reshape=True)

    print("==> loading test data")
    data_path = os.path.join(args.dataset_dir, 'test_data.npy')
    label_path = os.path.join(args.dataset_dir, 'test_label.pkl')
    valid_frame_path = os.path.join(args.dataset_dir, 'test_num_frame.npy')
    test_feeder = Feeder(data_path, label_path, valid_frame_path,
        normalization = args.normalization,
        ftrans = args.ftrans,
        reshape=True)
    return (train_feeder, test_feeder)

def get_parser():
    from parser import parser
    parser.add_argument('--weight_norm', dest='weight_norm', action='store_true')
    parser.add_argument('--no-weight_norm', dest='weight_norm', action='store_false')
    parser.set_defaults(weight_norm=True)
    return parser

def create_model():
    model = Autoencoder(
        input_size = input_size,
        norm = args.weight_norm
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    return (model, optimizer)

def load_model():
    print("==> loading existing lstm model")
    model_info = torch.load(model_path)
    model = Autoencoder(
        input_size = input_size,
        norm = args.weight_norm
    )
    #model.cuda()
    model.load_state_dict(model_info['state_dict'])
    best_loss = model_info['best_loss']
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    optimizer.load_state_dict(model_info['optimizer'])
    return (model, optimizer)

def save_loss(loss_list):
    loss_file_name = 'loss_' + args.version + '.npy'
    loss_file_path = os.path.join(args.checkpoint_folder, loss_file_name)
    np.save(loss_file_path, np.asarray(loss_list))

def draw_loss_fig():
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    loss_file_name = 'loss_' + args.version + '.npy'
    loss_file_path = os.path.join(args.checkpoint_folder, loss_file_name)
    loss = np.load(loss_file_path)
    fig_name = 'loss_' + args.version + '.png'
    fig_path = os.path.join(args.figure_folder, fig_name)
    from visualizer import draw_loss
    draw_loss(loss, save=fig_path)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_train_loss = 0.
    for batch_id, (data, target, frame_num) in enumerate(train_loader):
        data, target, frame_num = \
            data.float().cuda(), target.long().cuda(), frame_num.long().cuda()
        output = model(data)
        loss = criterion(output, target)
        total_train_loss += loss
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step() 

    train_loss = total_train_loss / len(train_loader.dataset)
    print('Training: Epoch:{:>3}, Total loss: {:.4f}'.format(epoch,
    total_train_loss))
    return total_train_loss

def train_model(model, train_loader, optimizer, criterion):
    print("==> print model properties")
    print(model.__dict__)
    total_loss = 0.
    best_epoch = 1
    best_loss = 1e9
    train_loss_list = []

    print("==> training model")
    try:
        epoch = 1
        for i in range(args.epochs):
            start = time.time()
            train_loss = train(model, train_loader, criterion, optimizer, epoch)

            elapsed_time = time.time() - start
            s = time.strftime("%M:%S", time.gmtime(elapsed_time))
            print("Time    :    {}".format(s)) 
            train_loss_list.append(train_loss)

            log("Epoch {:3d}, Train Loss {:.4f}".format(epoch, train_loss))

            if train_loss < best_loss:
                best_epoch = epoch
                best_loss = train_loss
                save_model({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict()})
            
            epoch += 1
    finally:
        # save and draw acc and loss curve
        save_loss(train_loss_list)
        draw_loss_fig()
        log("Best Epoch {:3d}, Train Loss {:.4f}".format(best_epoch, best_loss))

def use_model(model, feeders):
    parts = ['train', 'test']
    for part, feeder in zip(parts,feeders):
        total_frame = feeder.total_frame
        hidden_data = np.empty((total_frame, 20))
        recovered_data = np.empty((total_frame, input_size))
        hidden_path = os.path.join(args.dataset_dir, part + '_data_hidden.npy')
        recovered_path = os.path.join(args.dataset_dir, part + '_data_recovered.npy')

        for i in tqdm(range(total_frame)):
            input_data = torch.from_numpy(feeder[i][0])
            hidden = model.get_hidden(input_data).numpy()
            recovered= model(input_data).numpy()
            hidden_data[i] = hidden
            recovered_data[i] = recovered

        feeder.reset_data(hidden_data)
        feeder.separate_time_axis()
        np.save(hidden_path, hidden_data)

        feeder.reset_data(recovered_data)
        feeder.separate_time_axis()
        np.save(recovered_path, recovered_data)
            
if __name__=='__main__':
    # parse args
    parser = get_parser()
    args = parser.parse_args()

    # checkpoint path
    model_name =  'model_' + args.version + '.pt'
    model_path = os.path.join(args.checkpoint_folder, model_name)
    log_name = 'log_' + args.version + '.txt'
    log_path = os.path.join(args.checkpoint_folder, log_name)
    with open(log_path, "w+") as f:
        pass # clear log

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(args.gpus)

    # define loss and optimizer
    criterion = nn.MSELoss()
    criterion.cuda()

    # load or create model
    if args.resume:
        # load dataset
        train_feeder, test_feeder = load_input_data()
        input_size = train_feeder.feature_dim
        model, optimizer = load_model()
        use_model(model, [train_feeder, test_feeder])
    else:
        # load dataset
        train_loader = load_train_data()
        input_size = train_loader.dataset.feature_dim

        model, optimizer = create_model()
        train_model(model, train_loader, optimizer, criterion)

