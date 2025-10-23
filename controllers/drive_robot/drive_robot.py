import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
import torch.multiprocessing as mp
from controller import Robot
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import random 
import numpy as np
import csv
from scipy.spatial.transform import Rotation as Rot
from robot_controller import RobotController

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")
        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure

def run_backend(states, keyframes):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return
    if mode == Mode.RELOC:
        frame = states.get_frame()
        success = relocalization(frame, keyframes, factor_graph, retrieval_database)
        if success:
            states.set_mode(Mode.TRACKING)
        states.dequeue_reloc()
        return
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0]
    if idx == -1:
        return
    # Graph Construction
    kf_idx = []
    # k to previous consecutive keyframes
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]
    retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,
        k=config["retrieval"]["k"],
        min_thresh=config["retrieval"]["min_thresh"],
    )
    kf_idx += retrieval_inds

    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list
    frame_idx = [idx] * len(kf_idx)
    if kf_idx:
        factor_graph.add_factors(
            kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
        )

    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    if config["use_calib"]:
        factor_graph.solve_GN_calib()
    else:
        factor_graph.solve_GN_rays()

    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)
            print("Finished global optimization for kf ", idx)

def compute_position(vec):
    tx, ty, tz, qx, qy, qz, qw, s = vec
    t = np.array([tx, ty, tz])
    quat = np.array([qx, qy, qz, qw])
    rot = Rot.from_quat(quat)
    R = rot.as_matrix()
    pos = - (1 / s) * (R.T @ t)
    return pos

def establish_plane(poses):
    points_list = []
    for p in poses:
        vec = p[3:11]  # tx, ty, tz, qx, qy, qz, qw, s
        pos = compute_position(vec)
        points_list.append(pos)
    points = np.array(points_list)
    center = np.mean(points, axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    normal = eigenvectors[:, 2]
    variance_explained = sum(eigenvalues[:2]) / sum(eigenvalues)
    principal_components = eigenvectors[:, :2]
    return center, principal_components, normal, variance_explained

def project_position(pos, center, principal_components):
    centered = pos - center
    proj = centered @ principal_components
    return proj[0], proj[1]

if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False   # Changed to True to ensure images are saved
    datetime_now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')

    # Hardcode defaults
    dataset_path = "webots"
    config_path = "config/base.yaml"
    no_viz = True

    # Initialize Webots robot and camera
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    camera = robot.getDevice("Astra rgb")
    camera.enable(timestep)
    motor_l = robot.getDevice("motor_1")
    motor_r = robot.getDevice("motor_2")
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    imu = robot.getDevice("imu")
    imu.enable(timestep)

    motor_l.setPosition(float('inf'))
    motor_l.setVelocity(0.0)
    motor_r.setPosition(float('inf'))
    motor_r.setVelocity(0.0)

    max_speed = torch.pi * 2

    mobility = RobotController(robot, motor_l, motor_r, timestep)

    robot.step(timestep)

    load_config(config_path)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, no_viz)
    viz2main = new_queue(manager, no_viz)

    dataset = load_dataset(dataset_path, robot=robot, camera=camera)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model) 

    i = 0
    fps_timer = time.time()

    frames = []
    poses = []
    movement_types = ["forward"]  # Only forward movement

    save_interval = 100 # save point cloud and poses every 100 frames

    # Create directories for intermediate outputs
    base_dir = pathlib.Path(f"logs/{datetime_now}")
    pose_dir = base_dir / "poses"
    image_dir = base_dir / "images"
    pose_dir.mkdir(exist_ok=True, parents=True)
    image_dir.mkdir(exist_ok=True, parents=True)

    pose_csv = pose_dir / "poses_log.csv"
    with open(pose_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp", "movement",
            *[f"T_WC_{j}" for j in range(8)],
            "gps_x", "gps_y", "gps_z",
            "imu_roll", "imu_pitch", "imu_yaw"
        ])

    buffer = []

    initial_exploration_steps = 50
    plane_established = False
    center = None
    principal_components = None

    while True:
        # print(f"Processing frame {i}")
        if robot.step(timestep) == -1:
            print("Webots simulation stopped")
            states.set_Mode(Mode.TERMINATED)
            break

        # Default to forward movement
        movement = "forward"
        motor_l.setVelocity(max_speed)
        motor_r.setVelocity(max_speed)

        # Step for the given velocities 
        for _ in range(5):
            if robot.step(timestep) == -1:
                print("Webots simulation stopped")
                states.set_Mode(Mode.TERMINATED)
                break

        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )

        # Extract pose data
        try:
            pose_data = T_WC.vec().cpu().numpy()  # Try using .vec() method
        except AttributeError:
            # Fallback: extract quaternion and translation manually
            translation = T_WC.trans.cpu().numpy()
            rotation = T_WC.quat.cpu().numpy()
            pose_data = np.concatenate([rotation, translation])
            print(f"\tExtracted pose data manually (quat + trans): {pose_data}")

        gps_values = gps.getValues()
        imu_values = imu.getRollPitchYaw()

        # Append collected data
        entry = [
            i,
            timestamp,
            movement,
            *np.ravel(pose_data),
            *gps_values,
            *imu_values
        ]
        poses.append(entry)
        buffer.append(entry)

        # Print structured info with tabs for readability
        print(
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Frame {i}\n"
            f"\tTimestamp:\t{timestamp}\n"
            f"\tMovement:\t{movement}\n"
            f"\tT_WC:\t\t{np.array2string(np.round(pose_data, 4), separator=', ', suppress_small=True)}\n"
            f"\tGPS:\t\t{np.round(gps_values, 4)}\n"
            f"\tIMU:\t\t{np.round(imu_values, 4)}"
        )

        # Save intermediate results every `save_interval` frames
        if i > 0 and i % save_interval == 0:
            # Append to master CSV
            with open(pose_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(buffer)
            buffer.clear()
            print(f"\tAppended last {save_interval} entries to {pose_csv}")

            # Save intermediate CSV snapshot
            intermediate_csv = pose_dir / f"poses_{i}.csv"
            with open(intermediate_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame", "timestamp", "movement",
                    *[f"T_WC_{j}" for j in range(8)],
                    "gps_x", "gps_y", "gps_z",
                    "imu_roll", "imu_pitch", "imu_yaw"
                ])
                writer.writerows(poses)
            print(f"\tIntermediate CSV snapshot saved to {intermediate_csv}")

            # Save point cloud
            ply_file = base_dir / f"point_cloud_{i}.ply"
            eval.save_reconstruction(base_dir, ply_file.name, keyframes, last_msg.C_conf_threshold)
            print(f"\tIntermediate point cloud saved to {ply_file}")

        # Save image
        image_path = image_dir / f"frame_{i:05d}.png"
        camera.saveImage(str(image_path), 100)

        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)

        run_backend(states, keyframes)

        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")

        if not plane_established and i >= initial_exploration_steps:
            center, principal_components, normal, var_exp = establish_plane(poses[:initial_exploration_steps])
            print(f"Established plane: normal {normal}, variance explained {var_exp}")
            plane_established = True

        if plane_established:
            current_vec = poses[-1][3:11]
            current_pos = compute_position(current_vec)
            proj_x, proj_y = project_position(current_pos, center, principal_components)
            stuck = mobility.update_position(current_pos)
            if stuck:
                # Optionally reset SLAM states or just continue
                print("[Main] Robot was stuck and has performed random sequence.")
                continue  # Skip SLAM update during recovery
            print(f"Current position in plane: ({proj_x}, {proj_y})")

        i += 1

    # Save final poses
    pose_file = pose_dir / "poses.txt"
    with open(pose_file, "w") as f:
        f.write("Frame,Timestamp,Movement,T_WC\n")
        for frame_id, timestamp, movement, pose in poses:
            pose_str = " ".join(map(str, pose))
            f.write(f"{frame_id},{timestamp},{movement},{pose_str}\n")
    print(f"Final poses saved to {pose_file}")

    # Save final point cloud
    ply_file = base_dir / "point_cloud.ply"
    eval.save_reconstruction(
        base_dir,
        ply_file.name,
        keyframes,
        last_msg.C_conf_threshold,
    )
    print(f"Final point cloud saved to {ply_file}")

    print("done")
    backend.join()
    if not no_viz:
        viz.join()