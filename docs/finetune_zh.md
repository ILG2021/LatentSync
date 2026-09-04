# LatentSync 微调指南（512 分辨率）

面向单卡 RTX 5090（32GB）的完整流程。所有命令在仓库根目录执行。

---

## 0. 环境

### 0.1 PyTorch

5090 是 Blackwell（sm_120），必须装 CUDA 12.8 编译的 PyTorch，**先装它再装其他依赖**，否则 `requirements.txt` 可能拽进 CPU 版或 cu121 版：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

期望 capability 为 `(12, 0)`。

### 0.2 bitsandbytes

两个 512 配置默认开启 `optimizer.use_8bit_adam`，把 AdamW 的两个动量从 fp32 量化到 8 位，**stage1 省约 7.1 GiB**。

```bash
pip install bitsandbytes
```

若报 sm_120 不支持，需要更新的版本；实在装不上，把 `configs/unet/stage1_512.yaml` 的 `optimizer.use_8bit_adam` 改为 `false`（显存需求回到约 30GB，32GB 卡仍能跑但余量很小）。

### 0.3 模型权重

```bash
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
```

```bash
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
```

```bash
huggingface-cli download stabilityai/sd-vae-ft-mse
```

只有 stage2 需要 SyncNet：

```bash
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
```

VAE 会在训练启动时从 HuggingFace 拉取，提前下好可避免跑到一半断网。

---

## 1. 素材要求

把原始视频放进 `my_data/raw/`（可以有子目录）。

| 项 | 要求 |
|---|---|
| 总时长 | 20~30 分钟起步，1~2 小时舒适 |
| 格式 | mp4 最佳；mov/mkv/avi/webm/flv/m4v/mpg/wmv/ts 等会自动转换 |
| 内容 | 单人正面说话，人脸清晰 |
| 覆盖度 | **比总时长更重要**——各种口型、语速、头部小幅转动 |

同一姿态录 2 小时，不如 30 分钟有变化的素材。

---

## 2. 数据预处理

```bash
python -m preprocess.data_processing_pipeline --total_num_workers 32 --per_gpu_num_workers 8 --resolution 512 --segment_seconds 5 --temp_dir temp --input_dir my_data/raw
```

Windows 用 `.\data_processing_pipeline.ps1`。

### 流水线的六步

```
my_data/raw
  ├─ 1. 归一化为 mp4    非 mp4 容器转码，原文件移到 my_data/converted_originals/
  ├─ 2. 隔离损坏文件    decord 打不开的（含无音轨）移到 my_data/broken/
  ├─ 3. 重采样          → my_data/resampled/   25fps / 16kHz，强制每 125 帧一个关键帧
  ├─ 4. 切片            → my_data/segmented/   严格 5.00 秒
  ├─ 5. 丢弃过短片段    不足 112 帧（4.5 秒）的尾巴被删除
  └─ 6. 人脸对齐裁剪    → my_data/affine_transformed/   512×512
```

每步产出独立目录，中断后重跑会跳过已完成的文件。

### 关键设计

**为什么强制关键帧间隔**：切片用 `-c:v copy`，只能在关键帧处切。不控制 GOP 的话 x264 默认 10 秒一个关键帧，`-segment_time 5` 实际会切出 10 秒的片段。第 3 步的 `-g 125 -keyint_min 125 -sc_threshold 0` 让切点精确落在 5 秒整。

**为什么丢尾巴**：`UNetDataset` 是**按视频等概率随机抽**，不按时长加权。一个 2.8 秒的尾巴和一个 5 秒的主段被抽中次数一样，那几十帧会被过采样。等长切分是保证采样均匀的唯一办法。

**人脸检测容错**：单帧检测失败不再丢弃整段——先检测整条轨迹，缺失的用左右邻居线性插值，**连续缺失超过 25 帧**才判定该片段不可用。

### 预处理后检查

```bash
find my_data/affine_transformed -name "*.mp4" | wc -l
```

```bash
ls my_data/broken/
```

被隔离的文件确认一下是真坏还是误判。`my_data/resampled` 和 `my_data/segmented` 是中间产物，确认无误后可删（但删了就没有断点续跑）。

---

## 3. 生成训练索引

```bash
python -m tools.write_fileslist
```

这一步做四件事：

1. 扫描 `my_data/affine_transformed`，把所有片段汇成一个池
2. 从池里**随机抽 10 条**作为验证集，写入 `my_data/val_clips.txt` 固化
3. 其余**全部**写入 `my_data/fileslist.txt`
4. 按片段数算好 `max_train_steps`，连同验证路径写回三个 512 配置

### 关于验证集

抽 10 条只是为了有个稳定的观察对象——真正用于生成验证视频的是其中第 1 条，另外 9 条只是不参与训练。

这里**不做同源片段的隔离**。单人微调的目标本来就是让模型记住这个身份和这套光照，需要留出的只是没见过的语音内容和口型轨迹，而同一条录像切出的其他片段本来就是不同的句子。按整条源视频留出会白白丢掉大量数据——你 30~60 分钟一条的素材，一条就是 360~720 个片段。

### 步数公式

```
max_train_steps = ceil(片段数 × passes / (batch_size × num_processes))
向上取整到 500 的倍数，下限 2000
```

`passes` 在 `tools/write_fileslist.py` 的 `UNET_CONFIGS` 里：stage1 是 30，stage2 是 15。stage2 更少是因为主干冻结、要拟合的东西少，而每步多了 VAE 解码和三个损失网络。

| 素材总时长 | 片段 | stage1 步数 | 单卡耗时（约 1~2 秒/步） | stage2 步数 |
|---|---|---|---|---|
| 30 min | 360 | 11000 | 3~6 小时 | 5500 |
| 1 h | 720 | 22000 | 6~12 小时 | 11000 |
| 2 h | 1440 | 43500 | 12~24 小时 | 22000 |
| 4 h | 2880 | 86500 | 24~48 小时 | 43500 |

步数给的是"充分训练"的上限，不是必须跑完——中途的 checkpoint 随时可用，觉得够了就停。想缩短就把 `UNET_CONFIGS` 里的 passes 从 30 调到 15~20。

### 参数

```bash
python -m tools.write_fileslist --num_processes 1 --num_val_clips 10 --seed 1247
```

- `--num_processes`：**必须和 `train_unet.sh` 里的 `--nproc_per_node` 一致**。单卡保持 1。
- `--num_val_clips`：验证集片段数，默认 10。
- `--seed`：首次抽取的随机种子。之后由 `val_clips.txt` 固化，重跑不会变。

后续追加素材再跑这个命令，验证集保持不变。若其中某些片段被删了，会告警并从剩余池里补齐到 10 条，已有的不动。

### 预热音频缓存（可选）

`audio2feat` 是训练时按需生成缓存的，上千片段的第一轮会被 whisper 推理拖慢。提前灌满：

```bash
python -c "from latentsync.whisper.audio2feature import Audio2Feature; e = Audio2Feature(model_path='checkpoints/whisper/tiny.pt', device='cuda', audio_embeds_cache_dir='my_data/cache/embeds', num_frames=16, audio_feat_length=[2, 2]); [e.audio2feat(l.strip()) for l in open('my_data/fileslist.txt', encoding='utf-8')]"
```

---

## 4. Stage1 训练

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25679 -m scripts.train_unet --unet_config_path "configs/unet/stage1_512.yaml"
```

或直接 `./train_unet.sh` / `.\train_unet_stage1.ps1`。

### stage1 在做什么

纯 latent 空间，**只有重建损失**（`pixel_space_supervise: false` 让 LPIPS / TREPA / SyncNet 全部跳过），无时序层，**全参数训练**。目标是让模型适应你的人脸域和分辨率。

显存约 23 GB（含 8-bit Adam）。

### 首次试跑

正式开跑前建议先验证不会 OOM：把 `configs/unet/stage1_512.yaml` 的 `max_train_steps` 临时改成 200、`save_ckpt_steps` 改成 100，跑完看日志里的峰值。

跑完记得复原：`write_fileslist` 只会重算 `max_train_steps`，**`save_ckpt_steps` 得自己改回 5000**——留着 100 的话每 100 步就存一次并触发轮转，磁盘和 IO 都吃不消。

### 监控

```bash
tensorboard --logdir debug/unet
```

| 指标 | 看什么 |
|---|---|
| `train/reconstruction_loss` | stage1 唯一在动的损失，应稳定下降 |
| `train/peak_vram_gib` | 实测显存峰值，每次验证时记录并重置 |
| `debug/unet/stage1/train-*/val_videos/` | 验证视频，肉眼看嘴型最直接 |

日志里也会打印 `Peak VRAM so far: XX.X GiB`。

### 产物

```
debug/unet/stage1/train-<时间戳>/
  ├─ checkpoints/
  │    ├─ checkpoint-5000.pt                    权重（推理用这个，约 5 GB）
  │    └─ checkpoint-5000.training_state.pt     优化器/调度器状态（约 2.5 GB，仅续训用）
  ├─ val_videos/
  ├─ tensorboard/
  └─ stage1_512.yaml                            本次训练的配置快照
```

`.training_state.pt` 单独存是为了不让权重文件变大。想省磁盘可以只删它，不影响推理。

### 存档策略

```yaml
save_ckpt_steps: 5000     # 每 5000 步存一次
max_keep_ckpts: 10        # 只保留最新 10 个，更旧的连 sidecar 一起删除
```

两者相乘决定了你能回溯多远。默认组合是按"10 个存档点铺满整个训练过程"设计的：约 5 万步的训练正好每 5000 步一个、留满 10 个，磁盘占用约 **75 GB**。

如果你的数据量差很多导致 `max_train_steps` 明显偏离 5 万，按 `save_ckpt_steps ≈ max_train_steps / 10` 调整，否则要么存档点覆盖不到训练早期，要么根本不会触发轮转。

`max_keep_ckpts: 0` 表示全部保留。

### 中断续训

配置里的 `resume_ckpt_path` 保持不动重跑，会从 `checkpoints/latentsync_unet.pt` 重新开始。**若要从中断处继续**，手动把 `resume_ckpt_path` 指向 `debug/unet/stage1/train-*/checkpoints/checkpoint-<步数>.pt`，优化器状态会自动从旁边的 sidecar 恢复。

---

## 5. Stage2 训练

### 先认清显存

| 配置 | 需求 | 32GB 卡 |
|---|---|---|
| `stage2_512.yaml` | 约 55 GB | 跑不了 |
| `stage2_512_efficient.yaml` | 未实测，明显更低 | 需要试 |

`stage2_512.yaml` 的 55GB 里参数账只占 8.5 GiB，其余四十多 GB 是 VAE 带梯度解码、TREPA（VideoMAEv2）、LPIPS（VGG16）、SyncNet 的前向图。梯度检查点这四个模块**已经全部开启**，没有剩余余量。

`stage2_512_efficient.yaml` 砍掉了三处：

```yaml
trepa_loss_weight: 0                          # VideoMAEv2 完全不实例化，最大一刀
trainable_modules: [motion_modules., attn2.]  # 只训音频 cross-attn
motion_module_decoder_only: true              # 时序层只放 decoder
```

还不够的话，按顺序继续：`perceptual_loss_weight: 0`（砍 LPIPS），最后 `pixel_space_supervise: false`（退化为用重建损失训时序层，显存约等于 stage1）。

### 运行

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25679 -m scripts.train_unet --unet_config_path "configs/unet/stage2_512_efficient.yaml"
```

Windows PowerShell 可直接运行 `.\train_unet_stage2_efficient.ps1`。若显存足够并希望使用完整配置，则运行 `.\train_unet_stage2.ps1`。

### 自动衔接

两个 stage2 配置都是 `resume_ckpt_path: auto`，按两级回退查找：

1. `debug/unet/stage2/` 有 checkpoint，用自己的最新那个，**步数保留**（续训）
2. 没有，用 `debug/unet/stage1/` 的最新那个，**步数清零**（新阶段）
3. 两个都没有，报错

启动日志会说明走的哪条：

```
Resume checkpoint (resuming this stage): debug/unet/stage2/.../checkpoint-3000.pt
Resume checkpoint (initialising from the previous stage): debug/unet/stage1/.../checkpoint-11000.pt
```

排序规则是 `(run 目录名, 步数)`——run 目录名是启动时间戳，所以先比 run 新旧再比步数，不依赖文件时间戳。

### 监控

除 stage1 的指标外，多看：

| 指标 | 含义 |
|---|---|
| `train/sync_loss` | 唇音同步损失 |
| `train/perceptual_loss` | LPIPS |
| `validation/sync_confidence` | 验证视频的 SyncNet 置信度，**越高越好** |

stage2 通常不是跑满预算，而是**盯着 `validation/sync_confidence` 和验证视频手动叫停**。

---

## 6. 推理

```bash
python -m scripts.inference --unet_config_path "configs/unet/stage2_512.yaml" --inference_ckpt_path "debug/unet/stage1/train-<时间戳>/checkpoints/checkpoint-<步数>.pt" --inference_steps 20 --guidance_scale 1.5 --enable_deepcache --video_path "输入视频.mp4" --audio_path "输入音频.wav" --video_out_path "输出.mp4"
```

可调：`inference_steps` [20-50] 越高画质越好越慢；`guidance_scale` [1.0-3.0] 越高唇形越准但可能抖动。

### 只跑了 stage1 的重要提醒

`stage1_512.yaml` 是 `use_motion_module: false`，建模型时**根本没有时序层**，加载 1.6 权重时那部分被 `strict=False` 丢弃，存出的 checkpoint 也没有。

推理配置是 `use_motion_module: true`，加载时时序层权重缺失，保持零初始化状态。由于 `zero_initialize: true` 把输出投影置零，**时序分支输出恒为 0**——不报错、不崩溃，但 **1.6 的帧间一致性能力完全没有生效**。

嘴型贴合度不受影响。固定机位、头部动作幅度小的素材，这个代价通常可以接受；要拿回时序一致性必须跑完 stage2。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `CUDA out of memory` | 确认 `use_8bit_adam: true` 且 bitsandbytes 已装；看日志 `Peak VRAM` 定位 |
| `no kernel image is available` | PyTorch 不是 cu128 版，重装（见 0.1） |
| `bitsandbytes is not installed` | `pip install bitsandbytes`，或把 `use_8bit_adam` 改 `false` |
| `no checkpoint was found under ...` | stage2 的 `auto` 找不到 stage1 产物，先跑 stage1 |
| `Resuming at step N ... past max_train_steps` | 续训步数已超预算，调大 `max_train_steps` 或换 checkpoint |
| `SyncNet path is not provided` | stage2 缺 `checkpoints/stable_syncnet.pt` |
| 片段数远少于预期 | 看 `my_data/broken/`，以及预处理日志里的 `Discarded` / `Exception` 行 |
| 切出来不是 5 秒 | 确认走的是完整流水线；单独跑 `segment_videos` 而没先做重采样会失去关键帧对齐 |

---

## 8. 关键文件速查

| 路径 | 作用 |
|---|---|
| `my_data/raw/` | 放原始素材 |
| `my_data/affine_transformed/` | 训练用的 512 人脸片段 |
| `my_data/fileslist.txt` | 训练集索引 |
| `my_data/val_clips.txt` | 固化的 10 条验证片段 |
| `my_data/broken/` | 被隔离的损坏文件 |
| `my_data/converted_originals/` | 格式转换前的原文件 |
| `configs/unet/stage1_512.yaml` | stage1 配置 |
| `configs/unet/stage2_512.yaml` | stage2 完整版（需 ≥64GB） |
| `configs/unet/stage2_512_efficient.yaml` | stage2 低显存版 |
| `debug/unet/stage1/` `debug/unet/stage2/` | 各阶段训练产物 |
