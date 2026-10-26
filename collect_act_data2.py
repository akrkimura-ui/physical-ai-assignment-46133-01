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

    # ★ 高さ・奥行き感が明確に写る「斜め視点」へカメラ設定を変更
    CAM_POS = (0.75, 0.25, 0.65)
    cam = scene.add_camera(res=(224, 224), pos=CAM_POS, lookat=(0.41, 0.0, 0.05), fov=45.0)

    scene.build()
    end_effector = franka.get_link("hand")

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 8000, 8000], dtype=np.float32))
    franka.set_dofs_kv(np.array([450,  450,  350,  350,  200,  200,  200,  400,   400], dtype=np.float32))

    q_home = franka.inverse_kinematics(link=end_effector, pos=np.array([0.35, 0.0, 0.35]), quat=np.array([0.0, 1.0, 0.0, 0.0])).cpu().numpy()
    q_home_full = np.zeros(9, dtype=np.float32)
    q_home_full[:7] = q_home[:7]
    q_home_full[7:] = 0.04

    os.makedirs("act_dataset", exist_ok=True)
    np.save("act_dataset/q_home_full.npy", q_home_full)

    NUM_EPISODES = 50
    print(f"ACT用学習データの自動収集（中心把持・台座干渉防止軌道版）を開始します...")

    all_qpos, all_actions = [], []

    for ep in range(NUM_EPISODES):
        rand_cube_size = 0.04
        rand_plat_height = 0.08

        rand_cube_x = 0.42 + np.random.uniform(-0.015, 0.015)
        rand_cube_y = 0.12 + np.random.uniform(-0.015, 0.015)
        rand_plat_x = 0.40 + np.random.uniform(-0.015, 0.015)
        rand_plat_y = -0.12 + np.random.uniform(-0.015, 0.015)
        rand_cube_yaw = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)

        quat_cube_obj = euler_to_quaternion(0.0, 0.0, rand_cube_yaw)
        cube_pos_val = np.array([rand_cube_x, rand_cube_y, rand_cube_size / 2.0], dtype=np.float32)
        plat_pos_val = np.array([rand_plat_x, rand_plat_y, rand_plat_height / 2.0], dtype=np.float32)

        target_cube.set_pos(cube_pos_val)
        target_cube.set_quat(quat_cube_obj)
        target_platform.set_pos(plat_pos_val)

        franka.set_dofs_position(q_home_full)
        for _ in range(30):
            franka.control_dofs_position(q_home_full)
            scene.step()

        HAND_OFFSET = 0.108 # キューブの中心をしっかり深めに挟む高さ
        cube_center_z = rand_cube_size / 2.0
        
        pick_z = cube_center_z + HAND_OFFSET
        # ★ 台座への衝突を防ぐため、上面より 0.8cm 高い位置で優しくリリースする
        place_z = rand_plat_height + cube_center_z + HAND_OFFSET + 0.008

        quat_pick = euler_to_quaternion(np.pi, 0.0, -rand_cube_yaw)
        quat_place_rot = euler_to_quaternion(np.pi, 0.0, np.pi / 2.0)

        # 軌道生成 (IK)
        q_app_raw      = franka.inverse_kinematics(link=end_effector, pos=np.array([rand_cube_x, rand_cube_y, pick_z + 0.12]), quat=quat_pick).cpu().numpy()
        q_approach     = align_joint_angles(q_app_raw, q_home)

        q_prepick_raw  = franka.inverse_kinematics(link=end_effector, pos=np.array([rand_cube_x, rand_cube_y, pick_z + 0.02]), quat=quat_pick).cpu().numpy()
        q_prepick      = align_joint_angles(q_prepick_raw, q_approach)

        q_pick_raw     = franka.inverse_kinematics(link=end_effector, pos=np.array([rand_cube_x, rand_cube_y, pick_z]), quat=quat_pick).cpu().numpy()
        q_pick         = align_joint_angles(q_pick_raw, q_prepick)

        q_lift_raw     = franka.inverse_kinematics(link=end_effector, pos=np.array([rand_cube_x, rand_cube_y, place_z + 0.12]), quat=quat_pick).cpu().numpy()
        q_lift         = align_joint_angles(q_lift_raw, q_pick)

        q_trans_top    = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([rand_plat_x, rand_plat_y, place_z + 0.12]), quat=quat_pick).cpu().numpy(), q_lift)
        q_preplace_top = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([rand_plat_x, rand_plat_y, place_z + 0.12]), quat=quat_place_rot).cpu().numpy(), q_trans_top)
        
        q_preplace     = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([rand_plat_x, rand_plat_y, place_z + 0.02]), quat=quat_place_rot).cpu().numpy(), q_preplace_top)
        q_place        = align_joint_angles(franka.inverse_kinematics(link=end_effector, pos=np.array([rand_plat_x, rand_plat_y, place_z]), quat=quat_place_rot).cpu().numpy(), q_preplace)

        # 段階的なウェイポイント設計
        waypoints = [
            (q_approach,     True,  70),   # 1. 上空接近
            (q_prepick,      True,  40),   # 2. キューブ直上 2cm で位置合わせ（Pre-Pick）
            (q_pick,         True,  25),   # 3. 垂直に正確に降下して中心を捉える
            (q_pick,         False, 35),   # 4. 爪を閉じる
            (q_pick,         False, 20),   # 5. 接触力安定化
            (q_lift,         False, 50),   # 6. 持ち上げ
            (q_trans_top,    False, 90),   # 7. 台座上空へ移動
            (q_preplace_top, False, 30),   # 8. 回転合わせ
            (q_preplace,     False, 40),   # 9. 台座直上 2cm で一時停止（Pre-Place）
            (q_place,        False, 25),   # 10. 衝突しない高度へ減速着地
            (q_place,        True,  35),   # 11. 指を開いて設置
            (q_preplace_top, True,  90)    # 12. 垂直退避
        ]

        images, qpos_list, actions = [], [], []
        current_target_full = q_home_full.copy()

        for target_q, open_flag, steps in waypoints:
            next_target_full = np.zeros(9, dtype=np.float32)
            next_target_full[:7] = target_q[:7]
            next_target_full[7:] = 0.04 if open_flag else 0.0

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
            f.create_dataset("cube_pos", data=cube_pos_val)
            f.create_dataset("cube_quat", data=quat_cube_obj)
            f.create_dataset("plat_pos", data=plat_pos_val)

        print(f"Saved: act_dataset/episode_{ep:03d}.h5 (Yaw: {np.degrees(rand_cube_yaw):.1f} deg)")

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