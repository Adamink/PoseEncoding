# sys
import pickle

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
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 normalization=True,
                 ftrans=True,
                 label_minus_one=True
                 ):
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.normalization = normalization
        self.ftrans = ftrans
        self.init_joint_map()
        self.load_data()
        if label_minus_one:
            self.label_minus_one()
        self.normalize()

    def init_joint_map(self):
        self.joint_map = {'torso':1, 'left_hip': 13, 'right_hip': 17}
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

    def normalize(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        if self.ftrans:
            # new origin
            origin = np.mean(data_numpy[:, :, self.origins,:], axis = 2, keepdims= True)
            data_numpy = data_numpy - origin
            # normalize
            eps = 1e-8
            norm = np.linalg.norm(data_numpy, ord = 'fro', axis = (2,3), keepdims= True)
            data_numpy = data_numpy / (norm + eps)
            # new axis
            size = data_numpy.shape[0]
            max_frame = data_numpy.shape[1]
            for i in tqdm(range(size)):
                for j in range(max_frame):
                    x_axis = data_numpy[i,j, self.joint_map['left_hip'],:]
                    x_axis = x_axis / (np.linalg.norm(x_axis) + eps)
                    y_axis = data_numpy[i,j, self.joint_map['torso'],:]
                    y_axis = y_axis - x_axis * np.inner(y_axis,x_axis)
                    y_axis = y_axis / (np.linalg.norm(y_axis) + eps)
                    z_axis = np.cross(x_axis, y_axis)
                    z_axis = z_axis / (np.linalg.norm(z_axis) + eps)
                    data_numpy[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                    data_numpy[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                    data_numpy[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
            else:
                pass
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


def test(data_path, label_path, valid_frame_path, vid=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path, valid_frame_path),
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, frame_num = loader.dataset[index]
        data = np.transpose(np.reshape(data[0,:], (20,3)),(1,0)) 
        from visualizer import visualize
        visualize(data, False, '/home2/wuxiao/pose_encoding/figures/test_dataloader.png')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    plt.switch_backend('agg')
    data_path = "/home2/wuxiao/pose_encoding/dataset/test_data.npy"
    label_path = "/home2/wuxiao/pose_encoding/dataset/test_label.pkl"
    valid_frame_path = "/home2/wuxiao/pose_encoding/dataset/test_num_frame.npy"
    dataset = Feeder(data_path, label_path, valid_frame_path)
    print(np.bincount(dataset.label))
    print("min(label): %s" %str(np.min(dataset.label)))
    print("max(label): %s" %str(np.max(dataset.label)))
    test(data_path, label_path, valid_frame_path, vid='a09_s06_e02_v3_skeleton')