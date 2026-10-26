import os
import functools
import time
import numpy as np
import torch

# --- 1. Genesis 内部の torch.compile を C++ ビルド不要な aot_eager モードへ強制リダイレクト ---
_original_torch_compile = torch.compile

@functools.wraps(_original_torch_compile)
def _patched_torch_compile(*args, **kwargs):
    kwargs["backend"] = "aot_eager"
    return _original_torch_compile(*args, **kwargs)

torch.compile = _patched_torch_compile

import genesis as gs
from train_act import FastACTNetwork

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

def main():
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.1, 0.0, 0.75), camera_lookat=(0.4, 0.0, 0.1), camera_fov=45),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())

    # ダミーの初期位置でエンティティ生成
    target_cube = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.42, 0.12, 0.02)),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0))
    )
    target_platform = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.08), pos=(0.40, -0.12, 0.04)),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9, 0.8))
    )
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

    # collect_act_data2.py と完全同一のカメラ設定
    CAM_POS = (0.75, 0.25, 0.65)
    cam = scene.add_camera(res=(224, 224), pos=CAM_POS, lookat=(0.41, 0.0, 0.05), fov=45.0)

    scene.build()

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 8000, 8000], dtype=np.float32))
    franka.set_dofs_kv(np.array([450,  450,  350,  350,  200,  200,  200,  400,   400], dtype=np.float32))

    # 正規化パラメータの読み込み
    stats = np.load("act_dataset/norm_stats.npz")
    qpos_min, qpos_max = stats["qpos_min"], stats["qpos_max"]
    act_min, act_max = stats["action_min"], stats["action_max"]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    CHUNK_SIZE = 10
    policy = FastACTNetwork(action_dim=9, chunk_size=CHUNK_SIZE).to(device)
    policy.load_state_dict(torch.load("act_model.pth", map_location=device))
    policy.eval()

    q_home_full = np.load("act_dataset/q_home_full.npy")
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    episode_cnt = 0

    print("--- ACT 連続デプロイ評価開始 (Ctrl+C で終了) ---")

    try:
        while True:
            episode_cnt += 1

            # ★ エピソードごとにキューブ・台座の位置と回転をランダム化 (学習データ生成と同じ範囲)
            rand_cube_x = 0.42 + np.random.uniform(-0.015, 0.015)
            rand_cube_y = 0.12 + np.random.uniform(-0.015, 0.015)
            rand_plat_x = 0.40 + np.random.uniform(-0.015, 0.015)
            rand_plat_y = -0.12 + np.random.uniform(-0.015, 0.015)
            rand_cube_yaw = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)

            quat_cube = euler_to_quaternion(0.0, 0.0, rand_cube_yaw)

            target_cube.set_pos((rand_cube_x, rand_cube_y, 0.02))
            target_cube.set_quat(quat_cube)
            target_platform.set_pos((rand_plat_x, rand_plat_y, 0.04))

            # ロボット初期化
            franka.set_dofs_position(q_home_full)
            for _ in range(30):
                franka.control_dofs_position(q_home_full)
                scene.step()

            # Temporal Ensembling バッファの初期化
            all_time_actions = np.zeros((650, 650 + CHUNK_SIZE, 9), dtype=np.float32)
            k = 0.01

            print(f"\n[Episode {episode_cnt:03d}] 位置ランダム配置完了 (Yaw: {np.degrees(rand_cube_yaw):.1f} deg)")

            for t in range(600):
                rgb_img = cam.render(rgb=True)[0]
                rgb_img = np.ascontiguousarray(rgb_img[..., :3])

                img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float().to(device) / 255.0
                img_tensor = (img_tensor - mean) / std
                img_tensor = img_tensor.unsqueeze(0)

                current_q = franka.get_dofs_position()
                if isinstance(current_q, torch.Tensor): current_q = current_q.cpu().numpy()
                qpos_norm = 2.0 * (current_q - qpos_min) / (qpos_max - qpos_min + 1e-6) - 1.0
                qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_norm = policy(img_tensor, qpos_tensor).cpu().numpy()[0]
                    pred_actions = (pred_norm + 1.0) * 0.5 * (act_max - act_min + 1e-6) + act_min

                all_time_actions[t, t:t + CHUNK_SIZE] = pred_actions

                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()

                raw_action = np.sum(actions_for_curr_step * exp_weights[:, None], axis=0)

                franka.control_dofs_position(raw_action)
                scene.step()
                time.sleep(0.01)

            print(f"[Episode {episode_cnt:03d}] 完了")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n連続評価を終了します。")

if __name__ == "__main__":
    main()