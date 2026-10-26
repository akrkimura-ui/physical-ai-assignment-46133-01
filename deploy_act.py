import os
os.environ["TORCHDCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import time
import genesis as gs
import numpy as np
import torch
from collections import defaultdict
from train_act import FastACTNetwork

def main():
    # Apple Silicon (MPS) / CUDA / CPU の自動判定
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = FastACTNetwork().to(device)
    if not os.path.exists("act_model.pth") or not os.path.exists("act_dataset/norm_stats.npz"):
        print("モデルまたは統計量ファイルが見つかりません。")
        return

    model.load_state_dict(torch.load("act_model.pth", map_location=device))
    model.eval()

    stats = np.load("act_dataset/norm_stats.npz")
    qpos_min, qpos_max = stats["qpos_min"], stats["qpos_max"]
    act_min, act_max = stats["action_min"], stats["action_max"]
    q_home_full = np.load("act_dataset/q_home_full.npy")

    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.1, 0.0, 0.75), camera_lookat=(0.4, 0.0, 0.1), camera_fov=45),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())

    target_cube = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.45, 0.15, 0.02)),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0))
    )
    target_platform = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.08), pos=(0.40, -0.15, 0.04)),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9, 0.8))
    )
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

    CAM_POS = (0.42, 0.0, 0.95)
    cam = scene.add_camera(res=(128, 128), pos=CAM_POS, lookat=(CAM_POS[0], CAM_POS[1], 0.0), fov=45.0)

    scene.build()

    # ★ PDゲインの適正化（初期トルク・発振防止）
    franka.set_dofs_kp(np.array([1200, 1200, 1200, 1200, 500, 500, 500, 2000, 2000], dtype=np.float32))
    franka.set_dofs_kv(np.array([120,  120,  120,  120,  50,  50,  50,  200,  200], dtype=np.float32))

    CHUNK_SIZE = 10
    TEMPORAL_K = 0.01

    # ★ ResNet18用の標準化パラメータ
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    print("\n==================================================")
    print("  ACT (ResNet18・ローパスフィルタ・実時間同期版) 制御スタート")
    print("==================================================")

    for episode in range(1, 6):
        print(f"\n--- 【ACT 推論エピソード {episode}】 開始 ---")
        ###
        # ★ テスト用固定位置に置き換え
        rand_cube_size = 0.04
        rand_plat_height = 0.08
        rand_cube_x, rand_cube_y = 0.42, 0.12
        rand_plat_x, rand_plat_y = 0.40, -0.12

        target_cube.set_pos((rand_cube_x, rand_cube_y, rand_cube_size / 2.0))
        target_platform.set_pos((rand_plat_x, rand_plat_y, rand_plat_height / 2.0))
        #rand_cube_size = np.random.uniform(0.035, 0.055)
        #rand_plat_height = np.random.uniform(0.050, 0.110)

        # 物体と台座が衝突・干渉しないように 15cm 以上の離隔を確保
        #while True:
        #    rand_cube_x, rand_cube_y = np.random.uniform(0.34, 0.48), np.random.uniform(0.05, 0.22)
        #    rand_plat_x, rand_plat_y = np.random.uniform(0.34, 0.48), np.random.uniform(-0.22, -0.05)
        #    dist = np.linalg.norm([rand_cube_x - rand_plat_x, rand_cube_y - rand_plat_y])
        #    if dist >= 0.15:
        #        break

        #target_cube.set_pos((rand_cube_x, rand_cube_y, rand_cube_size / 2.0))
        #target_platform.set_pos((rand_plat_x, rand_plat_y, rand_plat_height / 2.0))

        # 初期姿勢セット時に制御命令を出し続けて静止させる
        franka.set_dofs_position(q_home_full)
        for _ in range(30):
            franka.control_dofs_position(q_home_full)
            scene.step()

        action_buffer = defaultdict(list)
        smoothed_target = q_home_full.copy()  # フィルタ用保持変数

        with torch.no_grad():
            for step in range(620):
                rgb_img = cam.render(rgb=True)[0]
                current_q = franka.get_dofs_position()
                if isinstance(current_q, torch.Tensor): current_q = current_q.cpu().numpy()

                qpos_norm = 2.0 * (current_q - qpos_min) / (qpos_max - qpos_min + 1e-6) - 1.0

                # 画像前処理（ResNet18用標準化の適用）
                img_tensor = torch.tensor(np.ascontiguousarray(rgb_img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(device)
                img_tensor = (img_tensor - mean) / std

                qpos_tensor = torch.tensor(np.ascontiguousarray(qpos_norm), dtype=torch.float32).unsqueeze(0).to(device)

                # 未来 10 コマのアクション推論
                pred_chunk_norm = model(img_tensor, qpos_tensor).cpu().numpy()[0]
                pred_chunk = 0.5 * (pred_chunk_norm + 1.0) * (act_max - act_min + 1e-6) + act_min

                for i in range(CHUNK_SIZE):
                    action_buffer[step + i].append(pred_chunk[i])

                # Temporal Ensembling (過去予測との指数重み付き平均)
                preds = np.array(action_buffer[step])
                weights = np.exp(-TEMPORAL_K * np.arange(len(preds))[::-1])
                weights /= weights.sum()

                raw_target = np.sum(preds * weights[:, None], axis=0)

                # 一次遅れローパスフィルタ (急激な目標変化をなだらかな応答にする)
                alpha = 0.85
                smoothed_target = (1.0 - alpha) * smoothed_target + alpha * raw_target

                franka.control_dofs_position(smoothed_target)
                scene.step()
                #time.sleep(0.01)  # 100Hzの実時間同期

                if step % 50 == 0:
                    err = np.linalg.norm(smoothed_target[:7] - current_q[:7])
                    print(f"Step {step:03d} | 関節ギャップ(L2): {err:.4f}")
            # ★ 3. 推論完了後のホールド（制御命令を出し続けて完全に離しきる）
            for _ in range(50):
                franka.control_dofs_position(smoothed_target)
                scene.step()

        print(f"エピソード {episode} 完了")
        #for _ in range(100): scene.step()

if __name__ == "__main__":
    main()