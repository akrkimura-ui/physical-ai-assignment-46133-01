import os
os.environ["TORCHDCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import genesis as gs
import numpy as np
import torch
import h5py

def euler_to_quaternion(roll, pitch, yaw):
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])

def align_joint_angles(q_target, q_ref):
    q_out = q_target.copy()
    for i in range(len(q_out)):
        diff = q_out[i] - q_ref[i]
        while diff > np.pi: q_out[i] -= 2 * np.pi
        while diff < -np.pi: q_out[i] += 2 * np.pi
    return q_out

def main():
    gs.init(backend=gs.cpu)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    target_cube = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04)), 
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0))
    )
    target_platform = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.08)), 
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9, 0.8))
    )
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

    CAM_POS = (0.42, 0.0, 0.95)
    cam = scene.add_camera(res=(128, 128), pos=CAM_POS, lookat=(CAM_POS[0], CAM_POS[1], 0.0), fov=45.0)

    scene.build()
    end_effector = franka.get_link("hand")

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 10000, 10000], dtype=np.float32))
    franka.set_dofs_kv(np.array([450,  450,  350,  350,  200,  200,  200,  500,   500], dtype=np.float32))

    # ★ ホーム初期姿勢の設定と保存
    q_home = franka.inverse_kinematics(link=end_effector, pos=np.array([0.35, 0.0, 0.35]), quat=np.array([0.0, 1.0, 0.0, 0.0])).cpu().numpy()
    q_home_full = np.zeros(9, dtype=np.float32)
    q_home_full[:7] = q_home[:7]
    q_home_full[7:] = 0.04

    os.makedirs("act_dataset", exist_ok=True)
    np.save("act_dataset/q_home_full.npy", q_home_full)

    NUM_EPISODES = 50
    print(f"ACT用学習データの自動収集（干渉回避・距離比例速度調整版）を開始します ({NUM_EPISODES} エピソード)...")

    all_qpos, all_actions = [], []

    for ep in range(NUM_EPISODES):
        ###
        rand_cube_size = 0.04
        rand_plat_height = 0.08
        rand_cube_x, rand_cube_y = 0.42, 0.12
        rand_plat_x, rand_plat_y = 0.40, -0.12

        target_cube.set_pos((rand_cube_x, rand_cube_y, rand_cube_size / 2.0))
        target_platform.set_pos((rand_plat_x, rand_plat_y, rand_plat_height / 2.0))
        #rand_cube_size = np.random.uniform(0.035, 0.055)
        #rand_plat_height = np.random.uniform(0.050, 0.110)

        # ★ 物体と台座が衝突・干渉しないように 15cm 以上の離隔を確保
        #while True:
            #rand_cube_x, rand_cube_y = np.random.uniform(0.34, 0.48), np.random.uniform(0.05, 0.22)   # 左側領域
            #rand_plat_x, rand_plat_y = np.random.uniform(0.34, 0.48), np.random.uniform(-0.22, -0.05) # 右側領域

            #dist = np.linalg.norm([rand_cube_x - rand_plat_x, rand_cube_y - rand_plat_y])
            #if dist >= 0.15:
                #break

        #target_cube.set_pos((rand_cube_x, rand_cube_y, rand_cube_size / 2.0))
        #target_platform.set_pos((rand_plat_x, rand_plat_y, rand_plat_height / 2.0))
        franka.set_dofs_position(q_home_full)
        for _ in range(30): scene.step()

        cube_pos = np.array([rand_cube_x, rand_cube_y, rand_cube_size])
        plat_pos = np.array([rand_plat_x, rand_plat_y, rand_plat_height])
        HAND_OFFSET = 0.098

        quat_pick = euler_to_quaternion(np.pi, 0.0, 0.0)
        quat_place_rot = euler_to_quaternion(np.pi, 0.0, np.pi / 2.0)

        q_approach  = franka.inverse_kinematics(link=end_effector, pos=np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15]), quat=quat_pick).cpu().numpy()
        q_pick      = franka.inverse_kinematics(link=end_effector, pos=np.array([cube_pos[0], cube_pos[1], (cube_pos[2] / 2.0) + HAND_OFFSET]), quat=quat_pick).cpu().numpy()
        q_lift      = franka.inverse_kinematics(link=end_effector, pos=np.array([cube_pos[0], cube_pos[1], max(plat_pos[2], cube_pos[2]) + 0.20]), quat=quat_pick).cpu().numpy()
        q_trans_top = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([plat_pos[0], plat_pos[1], plat_pos[2] + 0.20]), quat=quat_pick).cpu().numpy(), q_lift)
        q_place_top = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([plat_pos[0], plat_pos[1], plat_pos[2] + 0.20]), quat=quat_place_rot).cpu().numpy(), q_trans_top)
        q_place     = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([plat_pos[0], plat_pos[1], plat_pos[2] + (cube_pos[2] / 2.0) + HAND_OFFSET]), quat=quat_place_rot).cpu().numpy(), q_place_top)

        # ★ 移動距離に応じてステップ数を調整（長距離移動区間のステップ数を増やして急移動を防止／計500ステップ）
        waypoints = [
            (q_approach,  True,  80),   # ホームから接近（長距離）
            (q_pick,      True,  40),   # 下降（短距離）
            (q_pick,      False, 30),   # 把握ホールド
            (q_lift,      False, 50),   # 持ち上げ（中距離）
            (q_trans_top, False, 100),  # 台の上空へ移動（長距離）
            (q_place_top, False, 40),   # 向き微調整（短距離）
            (q_place,     False, 60),   # 設置（短距離）
            (q_place,     True,  60),   # 解放
            (q_place_top, True,  130)    # 退避（長距離）
        ]

        images, qpos_list, actions = [], [], []
        current_target_full = q_home_full.copy()

        for target_q, open_flag, steps in waypoints:
            next_target_full = np.zeros(9, dtype=np.float32)
            next_target_full[:7] = target_q[:7]
            next_target_full[7:] = 0.04 if open_flag else 0.0

            # 線形補間（Interpolation）でスムーズな関節運動を生成
            for s in range(1, steps + 1):
                alpha = s / float(steps)
                interp_action = (1.0 - alpha) * current_target_full + alpha * next_target_full

                rgb_img = cam.render(rgb=True)[0]
                current_q = franka.get_dofs_position()
                if isinstance(current_q, torch.Tensor): current_q = current_q.cpu().numpy()

                images.append(rgb_img)
                qpos_list.append(current_q)
                actions.append(interp_action)

                franka.control_dofs_position(interp_action)
                scene.step()

            current_target_full = next_target_full.copy()

        all_qpos.extend(qpos_list)
        all_actions.extend(actions)

        with h5py.File(f"act_dataset/episode_{ep:03d}.h5", "w") as f:
            f.create_dataset("images", data=np.array(images, dtype=np.uint8))
            f.create_dataset("qpos", data=np.array(qpos_list, dtype=np.float32))
            f.create_dataset("actions", data=np.array(actions, dtype=np.float32))

        print(f"Saved: act_dataset/episode_{ep:03d}.h5 ({len(images)} steps)")

    all_qpos = np.array(all_qpos)
    all_actions = np.array(all_actions)
    np.savez(
        "act_dataset/norm_stats.npz",
        qpos_min=all_qpos.min(axis=0), qpos_max=all_qpos.max(axis=0),
        action_min=all_actions.min(axis=0), action_max=all_actions.max(axis=0)
    )
    print("データ収集完了。")

if __name__ == "__main__":
    main()