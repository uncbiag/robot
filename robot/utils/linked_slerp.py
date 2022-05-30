import numpy as np
import pyvista as pv
from pyvista import examples

from scipy.spatial.transform import Rotation as R


def get_slerp_cam_pos(src_cam_pos, dst_cam_pos, t):
    # conver cam_pos to [R|T]
    # ref: https://www.songho.ca/opengl/gl_lookattoaxes.html
    # about fake up and real up vector
    # ref: https://gamedev.stackexchange.com/questions/139703/compute-up-and-right-from-a-direction
    def get_rt_matrix(cam_pos):
        forward = np.array(cam_pos[1]) - np.array(cam_pos[0])
        forward = forward / (np.sqrt(np.sum(forward ** 2)) + 1e-8)
        fake_up = np.array(cam_pos[2])
        left = np.cross(fake_up, forward)
        left = left / (np.sqrt(np.sum(left ** 2)) + 1e-8)
        up = np.cross(forward, left)
        up = up / (np.sqrt(np.sum(up ** 2)) + 1e-8)
        tr = np.array(cam_pos[0]) - np.array(cam_pos[1])
        rt_matrix = np.vstack([left, up, forward, tr]).T
        rt_matrix = np.vstack([rt_matrix, [0, 0, 0, 1]])
        return rt_matrix

    # only support spherical motion now
    assert src_cam_pos[1] == dst_cam_pos[1]
    assert np.linalg.norm(np.array(src_cam_pos[1]) - np.array(src_cam_pos[0])) - \
           np.linalg.norm(np.array(dst_cam_pos[1]) - np.array(dst_cam_pos[0])) < 1e-6
    radius = np.linalg.norm(np.array(src_cam_pos[1]) - np.array(src_cam_pos[0]))
    src_rt_matrix = get_rt_matrix(src_cam_pos)
    dst_rt_matrix = get_rt_matrix(dst_cam_pos)

    # check [R] matrix
    src_r_matrix = src_rt_matrix[:3, :3]
    assert np.sum(src_r_matrix @ src_r_matrix.T - np.eye(3)) < 1e-8
    dst_r_matrix = dst_rt_matrix[:3, :3]
    assert np.sum(dst_r_matrix @ dst_r_matrix.T - np.eye(3)) < 1e-8

    src_r = R.from_matrix(src_r_matrix)
    dst_r = R.from_matrix(dst_r_matrix)
    src_quat = src_r.as_quat()
    dst_quat = dst_r.as_quat()
    if np.sum(src_quat*dst_quat)<0:
        src_quat = -src_quat
    theta =np.arccos(np.sum(src_quat*dst_quat))
    sin_theta = np.sin(theta)
    cur_quat = (np.sin((1-t)*theta)*src_quat + np.sin((t)*theta)*dst_quat)/sin_theta
    cur_r = R.from_quat(cur_quat)
    cur_r_matrix = cur_r.as_matrix()
    cur_forward = cur_r_matrix[:, 0]
    cur_up = cur_r_matrix[:, 1]
    cur_left = cur_r_matrix[:, 2]

    src_vector = np.array(src_cam_pos[0]) - np.array(src_cam_pos[1])
    dst_vector = np.array(dst_cam_pos[0]) - np.array(dst_cam_pos[1])
    cur_src_vector = src_vector
    if np.sum(src_vector*dst_vector)<0:
        cur_src_vector = -src_vector
    theta = np.arccos(np.sum(cur_src_vector * dst_vector)/(np.linalg.norm(cur_src_vector,ord=2)*np.linalg.norm(dst_vector,ord=2)))
    sin_theta = np.sin(theta)
    cur_vector = (np.sin((1 - t) * theta) * src_vector + np.sin((t) * theta) * dst_vector) / sin_theta
    cur_center = (np.array(dst_cam_pos[1]) -np.array(src_cam_pos[1]))*t +np.array(src_cam_pos[1])
    cur_tr = cur_vector+ cur_center

    # src_tr = src_rt_matrix[:3, 3]
    # src_theta = np.arctan2(src_tr[2], src_tr[0])
    # src_phi = np.arctan2(src_tr[1], np.sqrt(src_tr[0] ** 2 + src_tr[2] ** 2))
    # dst_tr = dst_rt_matrix[:3, 3]
    # dst_theta = np.arctan2(dst_tr[2], dst_tr[0])
    # dst_phi = np.arctan2(dst_tr[1], np.sqrt(dst_tr[0] ** 2 + dst_tr[2] ** 2))
    # cur_theta = src_theta + (dst_theta - src_theta) * t
    # cur_phi = src_phi + (dst_phi - src_phi) * t
    # cur_tr = [radius * np.cos(cur_phi) * np.cos(cur_theta),
    #           radius * np.sin(cur_phi),
    #           radius * np.cos(cur_phi) * np.sin(cur_theta)]
    # cur_center = src_cam_pos[1]
    return [tuple(cur_tr), cur_center, tuple(cur_up)]



if __name__ == "__main__":
    pv.set_plot_theme("default")
    pv.create_axes_orientation_box()

    # download mesh
    mesh = examples.download_cow()

    decimated = mesh.decimate_boundary(target_reduction=0.75)

    p = pv.Plotter(notebook=0, shape=(1, 2), border=False)
    p.subplot(0, 0)
    p.add_text("Original mesh", font_size=24)
    p.add_mesh(mesh, show_edges=True, color=True)
    p.subplot(0, 1)
    p.add_text("Decimated version", font_size=24)
    p.add_mesh(decimated, color=True, show_edges=True)

    p.link_views()  # link all the views
    # Set a camera position to all linked views
    src_cam_pos = [(15, 5.0, 0),
                   (0, 0, 0),
                   (0, 1, 0)]
    # dst_cam_pos = [(8.387893552061202, 5.0, 12.435563588325627),
    #                (0.0, 0.0, 0.0),
    #                (0.0, 1.0, 0.0)]
    dst_cam_pos = [(0.0, 15.0, 5.0),
                   (0.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0)]
    p.camera_position = src_cam_pos

    get_slerp_cam_pos(src_cam_pos, dst_cam_pos, 0)

    p.open_gif("linked.gif")
    # Update camera and write a frame for each updated position
    nframe = 15
    for i in range(nframe):
        p.camera_position = get_slerp_cam_pos(src_cam_pos, dst_cam_pos, i / nframe)
        p.write_frame()

    # Close movie and delete object
    p.close()