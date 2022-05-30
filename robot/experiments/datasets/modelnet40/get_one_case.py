from robot.experiments.datasets.modelnet40.modelnet40_dataset_utils import *
from robot.global_variable import Shape
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate  %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def get_one_case(partitiion="train", num_points=1024,rot_factor=4, gaussian_noise=False,subsampled=False, num_subsampled_points=768,device= torch.device("cpu")):
    def get(item=0):
        data, label = load_data(partitiion)
        pointcloud = data[item][:num_points]
        np.random.seed(item)
        anglex = np.random.uniform() * np.pi / rot_factor
        angley = np.random.uniform() * np.pi / rot_factor
        anglez = np.random.uniform() * np.pi / rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points\
                (pointcloud1, pointcloud2, num_subsampled_points=num_subsampled_points)
        pointcloud1 = torch.tensor(pointcloud1[None]).transpose(2,1).contiguous().to(device)
        pointcloud2 = torch.tensor(pointcloud2[None]).transpose(2,1).contiguous().to(device)
        npoint = pointcloud1.shape[1]
        weight1 = torch.ones(1,npoint,1)/npoint
        weight2 = torch.ones(1,npoint,1)/npoint
        weight1, weight2 = weight1.to(device), weight2.to(device)
        source = Shape().set_data(points=pointcloud1, weights=weight1)
        target = Shape().set_data(points=pointcloud2, weights=weight2)
        return source, target
    return get

if __name__ == "__main__":
    goc = get_one_case()
    source_point, target_point = goc(item=0)






