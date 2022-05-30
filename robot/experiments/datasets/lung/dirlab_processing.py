import os,sys
sys.path.insert(0, os.path.abspath("../../../.."))
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from robot.datasets.data_utils import save_json
from robot.datasets.vtk_utils import read_vtk
from robot.shape.shape_utils import get_scale_and_center
from robot.experiments.datasets.lung.global_variable import lung_expri_path

"""
High resolution to low resoltuion dirlab mapping

1. flip the last dimension of the high resolution image
2. take the high resoltuion (max between the insp and exp )
3.  padding at the end

Landmark mapping
loc[z] = (high_image.shape[z] - low_index[z]*4 + 1.5)*high_spacing + high_origin ( 2 seems better in practice)

"""

COPD_ID={
    "copd1":  "copd_000001",
    "copd2":  "copd_000002",
    "copd3":  "copd_000003",
    "copd4":  "copd_000004",
    "copd5":  "copd_000005",
    "copd6":  "copd_000006",
    "copd7":  "copd_000007",
    "copd8":  "copd_000008",
    "copd9":  "copd_000009",
    "copd10": "copd_000010"
}

ID_COPD={
    "copd_000001":"copd1",
    "copd_000002":"copd2",
    "copd_000003":"copd3",
    "copd_000004":"copd4",
    "copd_000005":"copd5",
    "copd_000006":"copd6",
    "copd_000007":"copd7",
    "copd_000008":"copd8",
    "copd_000009":"copd9",
    "copd_000010":"copd10"
}

#in sitk coord
COPD_spacing = {"copd1": [0.625, 0.625, 2.5],
                "copd2": [0.645, 0.645, 2.5],
                "copd3": [0.652, 0.652, 2.5],
                "copd4": [0.590, 0.590, 2.5],
                "copd5": [0.647, 0.647, 2.5],
                "copd6": [0.633, 0.633, 2.5],
                "copd7": [0.625, 0.625, 2.5],
                "copd8": [0.586, 0.586, 2.5],
                "copd9": [0.664, 0.664, 2.5],
                "copd10": [0.742, 0.742, 2.5]}

# in sitk coord
COPD_low_shape = {"copd1": [512, 512, 121],
              "copd2": [512, 512, 102],
              "copd3": [512, 512, 126],
              "copd4": [512, 512, 126],
              "copd5": [512, 512, 131],
              "copd6": [512, 512, 119],
              "copd7": [512, 512, 112],
              "copd8": [512, 512, 115],
              "copd9": [512, 512, 116],
              "copd10":[512, 512, 135]}


# in sitk coord
COPD_high_insp_shape = {"copd1": [512, 512, 482],
              "copd2": [512, 512, 406],
              "copd3": [512, 512, 502],
              "copd4": [512, 512, 501],
              "copd5": [512, 512, 522],
              "copd6": [512, 512, 474],
              "copd7": [512, 512, 446],
              "copd8": [512, 512, 458],
              "copd9": [512, 512, 461],
              "copd10":[512, 512, 535]}


# in sitk coord
COPD_high_exp_shape = {"copd1": [512, 512, 473],
              "copd2": [512, 512, 378],
              "copd3": [512, 512, 464],
              "copd4": [512, 512, 461],
              "copd5": [512, 512, 522],
              "copd6": [512, 512, 461],
              "copd7": [512, 512, 407],
              "copd8": [512, 512, 426],
              "copd9": [512, 512, 380],
              "copd10":[512, 512, 539]}

COPD_info = {"copd1": {"insp":{'size': [512, 512, 482],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -310.625]},
                        "exp":{'size': [512, 512, 473],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -305.0]}},
              "copd2":  {"insp":{'size': [512, 512, 406],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-176.9, -165.0, -254.625]},
                        "exp":{'size': [512, 512, 378],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-177.0, -165.0, -237.125]}},
              "copd3":  {"insp":{'size': [512, 512, 502],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -343.125]},
                        "exp":{'size': [512, 512, 464],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -319.375]}},
              "copd4":  {"insp":{'size': [512, 512, 501],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -308.25]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -283.25]}},
              "copd5":  {"insp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]},
                        "exp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]}},
              "copd6":  {"insp":{'size': [512, 512, 474],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -299.625]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -291.5]}},
              "copd7":  {"insp":{'size': [512, 512, 446],'spacing': [0.625, 0.625, 0.625], 'origin': [-150.7, -160.0, -301.375]},
                        "exp":{'size': [512, 512, 407],'spacing': [0.625, 0.625, 0.625], 'origin': [-151.0, -160.0, -284.25]}},
              "copd8":  {"insp":{'size': [512, 512, 458],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -313.625]},
                        "exp":{'size': [512, 512, 426],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -294.625]}},
              "copd9":  {"insp":{'size': [512, 512, 461],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -310.25]},
                        "exp":{'size': [512, 512, 380],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -259.625]}},
              "copd10": {"insp":{'size': [512, 512, 535],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -355.0]},
                        "exp":{'size': [512, 512, 539],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -346.25]}}
              }





"""
before mapping
current COPD_ID;copd1 , and the current_mean 26.33421393688401                      current COPD_ID;copd1 , and the current_mean 26.33421393688401
current COPD_ID;copd2 , and the current_mean 21.785988375950623                     current COPD_ID;copd2 , and the current_mean 21.77096701290744
current COPD_ID;copd3 , and the current_mean 12.6391693237195                       current COPD_ID;copd3 , and the current_mean 12.641456423304232
current COPD_ID;copd4 , and the current_mean 29.583560337310402                     current COPD_ID;copd4 , and the current_mean 29.580001001346986
current COPD_ID;copd5 , and the current_mean 30.082670091996842                     current COPD_ID;copd5 , and the current_mean 30.066294774082003
current COPD_ID;copd6 , and the current_mean 28.456016850531874                     current COPD_ID;copd6 , and the current_mean 28.44935880947926
current COPD_ID;copd7 , and the current_mean 21.601714709640365                     current COPD_ID;copd7 , and the current_mean 16.04527530944317
current COPD_ID;copd8 , and the current_mean 26.456861641390127                     current COPD_ID;copd8 , and the current_mean 25.831153412715352
current COPD_ID;copd9 , and the current_mean 14.860263389215536                     current COPD_ID;copd9 , and the current_mean 14.860883966778562
current COPD_ID;copd10 , and the current_mean 21.805702262166907                    current COPD_ID;copd10 , and the current_mean 27.608698637477584
"""


CENTER_BIAS = 2

def read_dirlab(file_path, shape):
    dtype = np.dtype("<i2")
    fid = open(file_path, 'rb')
    data = np.fromfile(fid, dtype)
    img_np = data.reshape(shape)
    return img_np

def save_dirlab_into_niigz(file_path, output_path, fname, is_insp=False):
    img_np = read_dirlab(file_path, np.flipud(COPD_low_shape[fname]))
    img_sitk = sitk.GetImageFromArray(img_np)
    # img_sitk.SetOrigin()
    img_sitk.SetSpacing(np.array(COPD_spacing[fname]))
    if is_insp:
        saving_path = os.path.join(output_path,COPD_ID[fname]+"_INSP.nii.gz")
    else:
        saving_path = os.path.join(output_path,COPD_ID[fname]+"_EXP.nii.gz")
    sitk.WriteImage(img_sitk,saving_path)
    return saving_path




def read_landmark_index(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy list of positions.
    """

    with open(f_path) as fp:
        content = fp.read().split('\n')

        # Read number of points from second
        count = len(content) - 1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float32)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split('\t')
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])
        return points


def get_img_info(img_path):
    img = sitk.ReadImage(img_path)
    origin_itk = img.GetOrigin()
    spacing_itk = img.GetSpacing()
    img_shape_itk = img.GetSize()
    return img_shape_itk, spacing_itk, origin_itk


def transfer_landmarks_from_dirlab_to_high(dirlab_index, high_shape):
    new_index= dirlab_index.copy()
    new_index[:,-1] =(high_shape[-1]- dirlab_index[:,-1]*4) + CENTER_BIAS
    return new_index

def transfer_landmarks_from_dirlab_to_high_range(dirlab_index, high_shape):
    index_range_shape = list(dirlab_index.shape)+[2]
    new_index_range= np.zeros(index_range_shape)
    new_index_range[:,0,0] = dirlab_index[:,0] - 0.5
    new_index_range[:,0,1] = dirlab_index[:,0] + 0.5
    new_index_range[:,1,0] = dirlab_index[:,1] - 0.5
    new_index_range[:,1,1] = dirlab_index[:,1] + 0.5
    new_index_range[:,2,0] = (high_shape[-1] - dirlab_index[:,-1]*4) + CENTER_BIAS - 2
    new_index_range[:,2,1] = (high_shape[-1] - dirlab_index[:,-1]*4) + CENTER_BIAS + 2
    return new_index_range



def process_points(point_path, case_id, point_mapped_path, is_insp):
    index = read_landmark_index(point_path)
    copd = ID_COPD[case_id]
    phase = "insp" if is_insp else "exp"
    img_info = COPD_info[copd][phase]
    img_shape_itk, spacing_itk, origin_itk = np.array(img_info["size"]), np.array(img_info["spacing"]), np.array(img_info["origin"])
    downsampled_spacing_itk = np.copy(spacing_itk)
    downsampled_spacing_itk[-1] = downsampled_spacing_itk[-1]*4
    # downsampled_spacing_itk = COPD_spacing[ID_COPD[case_id]]
    print("spatial ratio corrections:")
    print("{} : {},".format(copd,np.array(COPD_spacing[copd])/downsampled_spacing_itk))
    transfered_index = transfer_landmarks_from_dirlab_to_high(index, img_shape_itk)
    transfered_index_range = transfer_landmarks_from_dirlab_to_high_range(index, img_shape_itk)
    physical_points = transfered_index*spacing_itk+origin_itk
    physical_points_range = transfered_index_range*(spacing_itk.reshape(1,3,1)) + origin_itk.reshape(1,3,1)
    # for i in range(len(physical_points)):
    #     physical_points[i][-1] = spacing_itk[-1] * img_shape_itk[-1] - index[i][-1]* COPD_spacing[ID_COPD[case_id]][-1] + origin_itk[-1]
    data = pv.PolyData(physical_points)
    data.point_arrays["idx"] = np.arange(1,301)
    data.save(point_mapped_path)
    np.save(point_mapped_path.replace(".vtk","_range.npy"), physical_points_range)
    name = os.path.split(point_mapped_path)[-1].split(".")[0]
    case_info = {"name":name,"raw_spacing":spacing_itk.tolist(), "raw_origin":origin_itk.tolist(),
                 "dirlab_spacing":downsampled_spacing_itk.tolist(), "raw_shape":img_shape_itk.tolist(), "z_bias":CENTER_BIAS, "dirlab_shape":COPD_low_shape[copd]}
    save_json(point_mapped_path.replace(".vtk","_info.json"), case_info)
    return physical_points






def get_center(point_path,case_id, is_insp):
    points = read_vtk(point_path)["points"]
    scale, center = get_scale_and_center(points, percentile=95)
    suffix = "_INSP_STD_USD_COPD" if is_insp else "_EXP_STD_USD_COPD"
    print('"{}":{}'.format(case_id+suffix,center[0]))


def compute_nn_dis_between_landmark_and_point_cloud(ldm_path, pc_path, case_id):
    import torch
    from robot.utils.knn_utils import KNN
    from robot.modules_reg.networks.pointconv_util import index_points_group
    landmark = read_vtk(ldm_path)["points"]
    raw_data_dict = read_vtk(pc_path)
    landmark[:,2] = landmark[:,2]+0.5
    pc_tensor = torch.Tensor(raw_data_dict["points"][None]).cuda()
    landmark_tensor = torch.Tensor(landmark[None]).cuda()
    knn = KNN(return_value=False)
    index = knn(landmark_tensor, pc_tensor,K=1)
    nn_pc = index_points_group(pc_tensor, index)
    diff = (landmark.squeeze() - nn_pc.squeeze().cpu().numpy())
    print("the current median shift of the case {} is {}".format(case_id, np.median(diff,0)))
    print("the current std shift of the case {} is {}".format(case_id, np.std(diff,0)))
    print("the current std shift of the case {} is {}".format(case_id,np.mean(np.linalg.norm(diff, ord=2, axis=1))))
    return diff

save_dirlab_IMG_into_niigz = False
project_landmarks_from_dirlab_to_high = True
compute_nearest_points_from_landmarks = False
visualize_landmarks_and_vessel_tree = False

copd_data_path = ""
if len(sys.argv):
    copd_data_path = sys.argv[1]
print(sys.argv)
#############  map dirlab landmarks into physical space  #########################
low_folder_path = copd_data_path
landmark_insp_key = "*_300_iBH_xyz_r1.txt"
processed_output_path = os.path.join(lung_expri_path,"dirlab_landmarks")
os.makedirs(processed_output_path,exist_ok=True)
id_list = list(ID_COPD.keys())
low_processed_folder = os.path.join(lung_expri_path, "dirlab_images")
os.makedirs(low_processed_folder, exist_ok=True)
if save_dirlab_IMG_into_niigz:
    low_img_insp_path_list = [os.path.join(low_folder_path,ID_COPD[fname],ID_COPD[fname],ID_COPD[fname]+"_iBHCT.img") for fname in id_list]
    low_img_exp_path_list = [os.path.join(low_folder_path,ID_COPD[fname],ID_COPD[fname],ID_COPD[fname]+"_eBHCT.img") for fname in id_list]
    os.makedirs(low_processed_folder, exist_ok=True)
    for low_img_insp_path, _id in zip(low_img_insp_path_list,id_list):
        save_dirlab_into_niigz(low_img_insp_path,low_processed_folder,ID_COPD[_id], is_insp=True)
    for low_img_exp_path, _id in zip(low_img_exp_path_list,id_list):
        save_dirlab_into_niigz(low_img_exp_path,low_processed_folder,ID_COPD[_id], is_insp=False)


landmark_dirlab_insp_path_list = [os.path.join(low_folder_path,ID_COPD[_id],ID_COPD[_id],ID_COPD[_id]+"_300_iBH_xyz_r1.txt") for _id in id_list]
landmark_dirlab_exp_path_list = [os.path.join(low_folder_path,ID_COPD[_id],ID_COPD[_id],ID_COPD[_id]+"_300_eBH_xyz_r1.txt") for _id in id_list]
landmark_physical_insp_path_list = [os.path.join(processed_output_path,_id+"_INSP.vtk") for _id in id_list]
landmark_physical_exp_path_list = [os.path.join(processed_output_path,_id+"_EXP.vtk") for _id in id_list]
if project_landmarks_from_dirlab_to_high:
    landmark_insp_physical_pos_list = [process_points(landmark_dirlab_insp_path_list[i], id_list[i],landmark_physical_insp_path_list[i], is_insp=True) for i in range(len(id_list))]
    landmark_exp_physical_pos_list = [process_points(landmark_dirlab_exp_path_list[i], id_list[i],landmark_physical_exp_path_list[i], is_insp=False) for i in range(len(id_list))]
    for path in landmark_physical_insp_path_list:
        assert os.path.isfile(path), "the file {} is not exist".format(path)
    for path in landmark_physical_exp_path_list:
        assert os.path.isfile(path), "the file {} is not exist".format(path)
    print("landmarks have been projected into physical space")


#############  analyze mapped results  #########################
"""compute the nearest distance between landmarks and vessel trees"""
pc_folder_path = copd_data_path
#pc_folder_path = "/home/zyshen/data/dirlab_data/DIRLABVascular_cleaned"
insp_key = "*INSP*"
shift_diff = []
pc_insp_path_list = [os.path.join(pc_folder_path,_id+"_INSP"+".vtk")  for _id in id_list]
pc_exp_path_list = [os.path.join(pc_folder_path,_id+"_EXP"+".vtk")  for _id in id_list]
if compute_nearest_points_from_landmarks:
    for i,_id in enumerate(id_list):
        insp_diff = compute_nn_dis_between_landmark_and_point_cloud(landmark_physical_insp_path_list[i],pc_insp_path_list[i],_id)
        exp_diff = compute_nn_dis_between_landmark_and_point_cloud(landmark_physical_exp_path_list[i],pc_exp_path_list[i],_id)
        shift_diff.append(insp_diff)
        shift_diff.append(exp_diff)
    shift_diff = np.concatenate(shift_diff,0)
    print("the current median shift of all dirlab cases is {}".format( np.median(shift_diff,0)))
    print("the current std shift of all dirlab cases is {}".format( np.std(shift_diff,0)))
    print("overall dist  of all dirlab mean {}".format(np.mean(np.linalg.norm(shift_diff, ord=2, axis=1))))


"""visualize mapped results"""
if visualize_landmarks_and_vessel_tree:
    from robot.utils.visualizer import visualize_landmark_pair, default_plot
    from robot.experiments.datasets.lung.visualizer import lung_plot, camera_pos
    for i, _id in enumerate(id_list):
        landmarks_insp = read_vtk(landmark_physical_insp_path_list[i])["points"]
        landmarks_exp = read_vtk(landmark_physical_exp_path_list[i])["points"]
        vessels_insp_dict = read_vtk(pc_insp_path_list[i])
        vessels_exp_dict = read_vtk(pc_exp_path_list[i])
        vessels_insp_points, vessels_exp_points = vessels_insp_dict["points"], vessels_exp_dict["points"]
        vessels_insp_weight, vessels_exp_weights = vessels_insp_dict["radius"], vessels_exp_dict["radius"]
        visualize_landmark_pair(
            vessels_exp_points,
            landmarks_exp,
            vessels_insp_points,
            landmarks_insp,
            vessels_exp_weights,
            np.ones_like(landmarks_exp),
            vessels_insp_weight,
            np.ones_like(landmarks_insp),
            "exp",
            "insp",
            point_plot_func=lung_plot(color='source'),
            landmark_plot_func=default_plot(cmap="Blues"),
            opacity=("linear", "linear"),
            light_mode="none",
        )










"""
num of 10 pair detected
origin copd1_insp:(-148.0, -145.0, -310.625)
size copd1_insp:(512, 512, 482)
spatial ratio corrections:
copd1 : [1. 1. 1.],
origin copd5_insp:(-145.9, -175.9, -353.875)
size copd5_insp:(512, 512, 522)
spatial ratio corrections:
copd5 : [1.00079816 1.00079816 1.        ],
origin copd8_insp:(-142.3, -147.4, -313.625)
size copd8_insp:(512, 512, 458)
spatial ratio corrections:
copd8 : [1.00010581 1.00010581 1.        ],
origin copd4_insp:(-124.1, -151.0, -308.25)
size copd4_insp:(512, 512, 501)
spatial ratio corrections:
copd4 : [1.00026448 1.00026448 1.        ],
origin copd6_insp:(-158.4, -162.0, -299.625)
size copd6_insp:(512, 512, 474)
spatial ratio corrections:
copd6 : [1.00029709 1.00029709 1.        ],
origin copd9_insp:(-156.1, -170.0, -310.25)
size copd9_insp:(512, 512, 461)
spatial ratio corrections:
copd9 : [0.99990664 0.99990664 1.        ],
origin copd2_insp:(-176.9, -165.0, -254.625)
size copd2_insp:(512, 512, 406)
spatial ratio corrections:
copd2 : [1.00072766 1.00072766 1.        ],
origin copd7_insp:(-150.7, -160.0, -301.375)
size copd7_insp:(512, 512, 446)
spatial ratio corrections:
copd7 : [1. 1. 1.],
origin copd3_insp:(-149.4, -167.0, -343.125)
size copd3_insp:(512, 512, 502)
spatial ratio corrections:
copd3 : [0.99947267 0.99947267 1.        ],
origin copd10_insp:(-189.0, -176.0, -355.0)
size copd10_insp:(512, 512, 535)
spatial ratio corrections:
copd10 : [0.99974669 0.99974669 1.        ],
origin copd1_exp:(-148.0, -145.0, -305.0)
size copd1_exp:(512, 512, 473)
spatial ratio corrections:
copd1 : [1. 1. 1.],
origin copd5_exp:(-145.9, -175.9, -353.875)
size copd5_exp:(512, 512, 522)
spatial ratio corrections:
copd5 : [1.00079816 1.00079816 1.        ],
origin copd8_exp:(-142.3, -147.4, -294.625)
size copd8_exp:(512, 512, 426)
spatial ratio corrections:
copd8 : [1.00010581 1.00010581 1.        ],
origin copd4_exp:(-124.1, -151.0, -283.25)
size copd4_exp:(512, 512, 461)
spatial ratio corrections:
copd4 : [1.00026448 1.00026448 1.        ],
origin copd6_exp:(-158.4, -162.0, -291.5)
size copd6_exp:(512, 512, 461)
spatial ratio corrections:
copd6 : [1.00029709 1.00029709 1.        ],
origin copd9_exp:(-156.1, -170.0, -259.625)
size copd9_exp:(512, 512, 380)
spatial ratio corrections:
copd9 : [0.99990664 0.99990664 1.        ],
origin copd2_exp:(-177.0, -165.0, -237.125)
size copd2_exp:(512, 512, 378)
spatial ratio corrections:
copd2 : [1.00072766 1.00072766 1.        ],
origin copd7_exp:(-151.0, -160.0, -284.25)
size copd7_exp:(512, 512, 407)
spatial ratio corrections:
copd7 : [1. 1. 1.],
origin copd3_exp:(-149.4, -167.0, -319.375)
size copd3_exp:(512, 512, 464)
spatial ratio corrections:
copd3 : [0.99947267 0.99947267 1.        ],
origin copd10_exp:(-189.0, -176.0, -346.25)
size copd10_exp:(512, 512, 539)
spatial ratio corrections:
copd10 : [0.99974669 0.99974669 1.        ],
current COPD_ID;copd1 , and the current_mean 26.33421393688401
current COPD_ID;copd2 , and the current_mean 21.77096701290744
current COPD_ID;copd3 , and the current_mean 12.641456423304232
current COPD_ID;copd4 , and the current_mean 29.580001001346986
current COPD_ID;copd5 , and the current_mean 30.066294774082003
current COPD_ID;copd6 , and the current_mean 28.44935880947926
current COPD_ID;copd7 , and the current_mean 16.04527530944317
current COPD_ID;copd8 , and the current_mean 25.831153412715352
current COPD_ID;copd9 , and the current_mean 14.860883966778562
current COPD_ID;copd10 , and the current_mean 27.608698637477584
average mean 23.31883032844186

Process finished with exit code 0

"""