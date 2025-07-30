# GenesisシミュレータにおけるUR5ロボットアームとRobotiq85グリッパーを用いた積み上げタスクの強化学習
## 実行環境
```
Ubuntu-24.04
```
## 環境構築
1. condaでの環境を作成
```
conda create -n genesis_world python=3.12
conda activate genesis_world
```
2. 依存関係のインストール
```
python3 -m pip install torch torchvision torchaudio
python3 -m pip install genesis-world libigl==2.5.1
```
3. 適切なバージョンのモーションプランナーをインストール
https://github.com/ompl/ompl/releases/tag/prerelease
```
wget https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.8.0-cp312-cp312-manylinux_2_28_x86_64.whl

python3 -m pip install ompl-1.8.0-cp312-cp312-manylinux_2_28_x86_64.whl
```

## 実行
### 実行スクリプト
- 環境変数の設定
```
export PYOPENGL_PLATFORM=glx
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:LD_LIBRARY_PATH
export MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA"
```
- 強化学習
```
python3 ur5_stack/ur5_stack_train.py
```
- 検証
```
python3 ur5_stack/ur5_stack_eval.py --ckpt 499
```
## その他Genesis公式チュートリアル
### genesisのクローン
```
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
```
### 強化学習テスト(Unitree Go2)

- 修正(examples/locomotion/go2.env)
```
-vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
+vis_options=gs.options.VisOptions(n_rendered_envs=1),

- rpy=True,
- degrees=True,
```
- 実行
```
pip install tensorboard
pip install rsl-rl-lib==2.2.4
python3 Genesis/examples/locomotion/go2_train.py
```
- 確認
```
python3 Genesis/examples/locomotion/go2_eval.py
```
### 強化学習テスト(Drone)
- 修正(examples/drone/hover_env.py)
```
-vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
+vis_options=gs.options.VisOptions(n_rendered_envs=1),

- rpy=True,
- degrees=True,
```
- 修正(engine/entities/drone_entity.py)
```
propellels_rpm = self.solver._process_dim(np.array(propellels_rpm.cpu(), dtype=gs.np_float)).T
```
- 実行
```
pip install tensorboard
pip install rsl-rl-lib==2.2.4 open3d
python3 Genesis/examples/drone/hover_train.py
```
- 確認
```
python3 Genesis/examples/drone/hover_eval.py
```
### tkinter
```
conda install tk
conda install example-robot-data -c conda-forge
```

### References
- [[Genesis] 1時間で4脚ロボットの強化学習の環境構築](https://qiita.com/hEnka/items/cc5fd872eb0bf7cd3abc)
- [Genesisで4脚ロボットMini Cheetahの強化学習](https://qiita.com/tamashu/items/3591a76d61e97fb3e0dc)
- [Genesis 入門 (5) - インバースキネマティクス と モーションプランニング](https://note.com/npaka/n/n3b06df2458c1)
- [GenesisでURロボット](https://natsutan.hatenablog.com/entry/2025/03/14/085329)
- https://x.com/kashu_yamazaki/status/1869519186311160021
- [Achieving Goals using Reward Shaping and Curriculum Learning](https://arxiv.org/abs/2206.02462)