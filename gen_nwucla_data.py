import numpy as np
import scipy.io as scio
import os
import pickle

from glob import glob
from numpy.lib.format import open_memmap

if __name__=='__main__':
    src_pth = '/data2/wuxiao/Northwestern-UCLA_skeleton/'
    dst_pth = '/home2/wuxiao/pose_encoding/dataset'
    parts = ['train', 'test']
    if not os.path.exists(dst_pth):
        os.mkdir(dst_pth)

    max_frame = 201
    num_joint = 20

    train_labels = []
    test_labels = []
    train_names = []
    test_names = []
    labels = {'train':train_labels, 'test':test_labels}
    names = {'train': train_names, 'test':test_names}

    for file_pth in glob(os.path.join(src_pth,'*')):
        basename = os.path.basename(file_pth)
        label = int(basename[1:3])
        data = np.asarray(scio.loadmat(file_pth)['skeleton'])
        # print(data.shape)
        if data.shape[0]!=20:
            continue
        if 'v3' in basename:
            test_labels.append(label)
            test_names.append(basename)
        else:
            train_labels.append(label)
            train_names.append(basename)

    for part in parts:
        with open('{}/{}_label.pkl'.format(dst_pth, part), 'wb') as f:
            pickle.dump((names[part], labels[part]), file = f)

        fp = open_memmap(
            '{}/{}_data.npy'.format(dst_pth, part),
            dtype='float32',
            mode='w+',
            shape=(len(labels[part]), max_frame, num_joint * 3))
        
        fl = open_memmap(
            '{}/{}_num_frame.npy'.format(dst_pth, part),
            dtype='int',
            mode='w+',
            shape=(len(labels[part]),))

        for i, basename in enumerate(names[part]):
            file_pth = os.path.join(src_pth, basename)
            data = np.asarray(scio.loadmat(file_pth)['skeleton'])
            # (20, 3, seq_len)
            data = np.transpose(data, axes = [2,0,1])
            # (seq_len, 20, 3)
            data = np.reshape(data, (data.shape[0], -1))
            try:
                fp[i, 0:data.shape[0], :] = data
            except:
                print(data.shape)
                print(names[part][i])
            fl[i] = data.shape[0]

            








