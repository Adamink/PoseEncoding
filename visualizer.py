import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
import os

def visualize(data, show = True, save = None):
    # data.shape: (3, 20)
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    paired_joints = [[7,8],[6,7],[5,6],[3,5],[3,4], [3,9],
        [9,10],[10,11],[11,12],[3,2],[2,1],[1,13],[13,14],[14,15],
        [15,16],[1,17],[17,18],[18,19],[19,20]]
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(data[0,:], data[1,:], data[2,:], c='r')
    for i in range(20):
        ax.text(data[0,i],data[1,i],data[2,i], str(i+1), size = 10)
    for pair in paired_joints:
        index = [x-1 for x in pair]
        x_start = data[0,index[0]]
        x_end = data[0,index[1]]
        y_start = data[1,index[0]]
        y_end = data[1,index[1]]
        z_start = data[2,index[0]]
        z_end = data[2,index[1]]
        ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end])
    # ax.plot([0,2],[1,3],[4,6], c = 'b')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)

if __name__=='__main__':
    src_pth = '/Users/wuxiao/Downloads/Northwestern-UCLA_skeleton/a08_s03_e03_v2_skeleton.mat'
    data = np.asarray(scio.loadmat(src_pth)['skeleton'])
    # (20, 3, seq_len) -> (3,20)
    data = np.transpose(data[:,:,0], axes = (1,0)) #(3,20)
    visualize(data)

