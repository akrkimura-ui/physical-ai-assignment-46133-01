import os
import glob
import time
import h5py
import numpy as np
import genesis as gs

def visualize_all_episodes():
    files = sorted(glob.glob("act_dataset/episode_*.h5"))
    if not files:
        print("データセットファイル (act_dataset/episode_*.h5) が見つかりません。")
        return

    print(f"全 {len(files)} 件のエピソードを順に再生します。\n")

    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.1, 0.0, 0.75), camera_lookat=(0.4, 0.0, 0.1), camera_fov=45),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())

    rand_cube_size = 0.04
    rand_plat_height = 0.08
    rand_cube_x, rand_cube_y = 0.42, 0.12
    rand_plat_x, rand_plat_y = 0.40, -0.12

    target_cube = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(rand_cube_x, rand_cube_y, rand_cube_size / 2.0)),
        surface=gs.surfaces.Default(color=(0.9, 0.2, 0.2, 1.0))
    )
    target_platform = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.08), pos=(rand_plat_x, rand_plat_y, rand_plat_height / 2.0)),
        surface=gs.surfaces.Default(color=(0.2, 0.5, 0.9, 0.8))
    )
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))

    scene.build()

    # ★ 必須: 指関節（インデックス 7, 8）へ確実に駆動力を伝えるためのPDゲイン設定
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 8000, 8000], dtype=np.float32))
    franka.set_dofs_kv(np.array([450,  450,  350,  350,  200,  200,  200,  400,   400], dtype=np.float32))

    for idx, file_path in enumerate(files):
        ep_num = int(os.path.basename(file_path).replace("episode_", "").replace(".h5", ""))
        
        with h5py.File(file_path, "r") as f:
            actions = np.array(f["actions"])
            qpos_list = np.array(f["qpos"])

        print(f"--- 【エピソード {ep_num:03d} 再生 ({idx + 1}/{len(files)})】 全 {len(actions)} ステップ ---")

        target_cube.set_pos((rand_cube_x, rand_cube_y, rand_cube_size / 2.0))
        target_platform.set_pos((rand_plat_x, rand_plat_y, rand_plat_height / 2.0))
        
        # 指が開いた状態（qpos_list[0]）を正しくセットして30ステップ安定化
        franka.set_dofs_position(qpos_list[0])
        for _ in range(30):
            franka.control_dofs_position(qpos_list[0])
            scene.step()

        for step, act in enumerate(actions):
            franka.control_dofs_position(act)
            scene.step()
            time.sleep(0.01)

        print(f"エピソード {ep_num:03d} 再生完了\n")
        time.sleep(0.5)

    print("全エピソードの再生が完了しました。")
    for _ in range(100):
        scene.step()

if __name__ == "__main__":
    visualize_all_episodes()