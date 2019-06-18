# sys
import pickle
import os
# torch
import torch
import numpy as np
import sys
from torch.utils import data
from tqdm import tqdm

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        normalization: If true, normalize input sequence
        ftrans: If true, apply normalization per frame; else per sequence
        reshape: merge the time axis into batch axis
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 normalization='default',
                 ftrans=True,
                 label_minus_one=True,
                 reshape=False
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.ftrans = ftrans
        self.eps = 1e-10
        self.init_joint_map()
        self.load_data()
        if label_minus_one:
            self.label_minus_one()
        if normalization!='none':
            self.origin_transfer()
            self.normalize()
            if normalization=='my':
                self.my_rotate()
            else:
                self.default_rotate()
        if reshape:
            self.merge_time_axis()
        
    def init_joint_map(self):
        self.joint_map = {'torso':1, 'left_hip': 13, 'right_hip': 17, 
         'shoulder_center':3, 'spine':2, 'left_shoulder':5, 'right_shoulder':9}
        self.origins = [1, 13, 17]
        # all minus one
        for joint in self.joint_map:
            self.joint_map[joint] = self.joint_map[joint] - 1
        for i in range(len(self.origins)):
            self.origins[i] = self.origins[i] - 1
            
    def load_data(self):
        # (batch, max_frame, feature)
        # (?, 201, 60)
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        
        # load data
        self.data = np.load(self.data_path)
        self.size, self.max_frame, self.feature_dim = self.data.shape
        # load num of valid frame length
        self.valid_frame_num = np.load(self.num_frame_path)
    
    def origin_transfer(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        origin = np.mean(data_numpy[:, :, self.origins,:], axis = 2, keepdims= True)
        data_numpy = data_numpy - origin
        self.data = np.reshape(data_numpy, (self.size, self.max_frame, self.feature_dim))

    def normalize(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        norm = np.linalg.norm(data_numpy, ord = 'fro', axis = (2,3), keepdims= True)
        data_numpy = data_numpy / (norm + self.eps)
        self.data = np.reshape(data_numpy, (self.size, self.max_frame, self.feature_dim))

    def my_rotate(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        dst = np.zeros_like(data_numpy)
        for i in tqdm(range(self.size)):
            for j in range(self.max_frame):
                shoulder_center = data_numpy[i,j,self.joint_map['shoulder_center'],:]
                spine = data_numpy[i,j,self.joint_map['spine'], :]
                right_shoulder = data_numpy[i,j, self.joint_map['right_shoulder'],:]
                left_shoulder = data_numpy[i,j, self.joint_map['left_shoulder'],:]

                new_z = shoulder_center - spine
                unit_z = new_z / np.linalg.norm(new_z + self.eps)
                new_x = right_shoulder - left_shoulder
                new_x = new_x - unit_z * np.inner(new_x, unit_z) 
                unit_x = new_x / np.linalg.norm(new_x + self.eps)
                unit_y = - np.cross(unit_x, unit_z)
                unit_y = unit_y / np.linalg.norm(unit_y + self.eps)
                x_axis = unit_x
                y_axis = unit_y
                z_axis = unit_z
                dst[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                dst[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                dst[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
        self.data = np.reshape(dst, (self.size, self.max_frame, self.feature_dim))
    def default_rotate(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        dst = np.zeros_like(data_numpy)
        for i in tqdm(range(self.size)):
            for j in range(self.max_frame):
                x_axis = data_numpy[i,j, self.joint_map['left_hip'],:]
                x_axis = x_axis / (np.linalg.norm(x_axis) + self.eps)
                y_axis = data_numpy[i,j, self.joint_map['torso'],:]
                y_axis = y_axis - x_axis * np.inner(y_axis,x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + self.eps)
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / (np.linalg.norm(z_axis) + self.eps)
                dst[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                dst[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                dst[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
                    
        self.data = np.reshape(dst, (self.size, self.max_frame, self.feature_dim))

    # to implement
    def quick_rotate(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        x_axis = data_numpy[:,:,self.joint_map['left_hip'],:]
        x_axis = x_axis / (np.linalg.norm(x_axis, axis = 2) + self.eps)
        y_axis = data_numpy[:,:,self.joint_map['torso'],:]
        y_axis = y_axis - np.einsum('ijk,ij->ijk',x_axis, np.einsum('ijk,ijk->ij', y_axis, x_axis))
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis, axis = 2) + self.eps)  
        M = np.stack([x_axis,y_axis,z_axis], axis = 2) # (size, max_frame, 3(stack), 3)
        #(size, max_frame, joint, 3)
        #(size,max_frame, joint, 3)
        data_numpy = np.einsum('ijkt,ijlk->ijlk', M, data_numpy)
        for i in tqdm(range(self.size)):
            for j in range(self.max_frame):
                x_axis = data_numpy[i,j, self.joint_map['left_hip'],:]
                x_axis = x_axis / (np.linalg.norm(x_axis) + self.eps)
                y_axis = data_numpy[i,j, self.joint_map['torso'],:]
                y_axis = y_axis - x_axis * np.inner(y_axis,x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + self.eps)
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / (np.linalg.norm(z_axis) + self.eps)
                dst[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                dst[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                dst[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
                    
        self.data = np.reshape(data_numpy, (self.size, self.max_frame, self.feature_dim))
    
    def label_minus_one(self):
        self.label = [x - 1 for x in self.label]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]
        label = self.label[index]
        valid_frame_num = self.valid_frame_num[index]
        return data_numpy, label, valid_frame_num

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def merge_time_axis(self):
        self.total_frame = np.sum(self.valid_frame_num)
        ret = np.empty((self.total_frame, self.feature_dim))
        index = 0
        for i in range(self.size):
            frames = self.valid_frame_num[i]
            ret[index:index+frames,:] = np.reshape(
             self.data[i, :frames, :], (frames, self.feature_dim))
            index = index + frames
        self.data = ret

    def separate_time_axis(self):
        ret = np.zeros((self.size, self.max_frame, self.feature_dim))
        index = 0
        for i in range(self.size):
            frames = self.valid_frame_num[i]
            ret[i, :frames, :] = self.data[index:index+frames, :]
            index = index + frames
        self.data = ret
    
    def reset_data(self, data):
        self.data = data
        self.feature_dim = data.shape[-1]
        
def test(data_path, label_path, valid_frame_path, vid=None, local=True):
    import matplotlib.pyplot as plt
    norm = 'default' if args.modality=='' else 'none'
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path, valid_frame_path, normalization = norm),
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, frame_num = loader.dataset[index]
        data = np.transpose(np.reshape(data[0,:], (20,3)),(1,0)) # (3, 20)
        from visualizer import visualize
        if not local:
            import matplotlib.pyplot as plt
            plt.switch_backend('agg')
            visualize(data, False, './figures/{}{}.png'.format(vid, args.modality))
        else:
            visualize(data, True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, choices=['train','test'], default='train')
    parser.add_argument('--modality', type=str, choices=['','_recovered','_hidden'], default='')
    parser.add_argument('--vid', type=str, default='a01_s01_e00_v2_skeleton')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument('--local', dest='local', action='store_true')
    parser.set_defaults(local=False)
    args = parser.parse_args()
    data_path = os.path.join(args.dataset_dir, args.part + '_data' + args.modality + '.npy')
    label_path = os.path.join(args.dataset_dir, args.part + '_label.pkl')
    valid_frame_path = os.path.join(args.dataset_dir, args.part + '_num_frame.npy')

    # dataset = Feeder(data_path, label_path, valid_frame_path, normalization = False)
    test(data_path, label_path, valid_frame_path, vid=args.vid, local=False)