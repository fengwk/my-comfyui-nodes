# my-comfyui-nodes

个人 ComfyUI 自定义节点仓。算法放在 `my_nodes/core/`，节点接线放在 `my_nodes/nodes/`，注册表集中在 `my_nodes/registry.py`。

## 安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fengwk/my-comfyui-nodes.git
# 或: ln -s /path/to/my-comfyui-nodes /path/to/ComfyUI/custom_nodes/my-comfyui-nodes
```

依赖：`numpy`、`Pillow`、`einops`、`safetensors`（Comfy 环境一般已有）。流式节点 `My Video Enhance Stream` 另外要求
`PATH` 上有 FFmpeg 5.1+ 的 `ffmpeg` 和 `ffprobe`（找不到时直接报缺失的工具名）。
IMAGE 输出的 RAM 预检使用 ComfyUI 已有依赖 `psutil`，无需额外安装。

可选：`vendor/sol_attn_minimax_v2.py` 需要 comfy-kitchen 的 `sol_attn` 内核
（Kijai `sol_attn` 分支构建，官方 PyPI 包不含）。缺失时该节点不注册，其余节点不受影响。

## IMAGE 工作流的输入约定：用 IMAGE 批次，不要图片列表

本节只适用于 IMAGE 流程。原生 `VIDEO` 输入不拼 IMAGE 批次，走
[`My Video Enhance Stream`](#my-video-enhance-stream)。

Comfy 的视频帧标准类型是 `IMAGE`：形状 `[N, H, W, C]`、float32、`[0, 1]`。

`GetVideoComponents` 输出的就是这个；`CreateVideo` / 多数图像节点吃的也是这个。  
**不要**做成 Python `list[Image]`——Comfy 没有一等公民的“图片列表”类型，下游节点接不上。

推荐接法：

```text
LoadVideo
  → GetVideoComponents.images     # IMAGE [N,H,W,C]
    → MiniMax H3 Inject Tail Noise    # 仍输出 IMAGE
      → CreateVideo               # 再交给 ChainExternalVideo / SaveVideo
```

不要把已经注过噪的输出再拿去抽下一段干净 context。

## 节点

### MiniMax H3 Inject Tail Noise

把 T3 紫绿色块灌进 IMAGE 批次的尾帧。配方默认：`tail=22`，前 19 帧 `alpha=0.45`，末 3 帧渐到 `0.10`。

| 输入 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `images` | IMAGE | — | 整段帧批次 |
| `tail_frames` | INT | 22 | 从末尾往前注多少帧 |
| `alpha` | FLOAT | 0.45 | 尾部前段混合强度 |
| `alpha_end` | FLOAT | 0.10 | 最后一帧强度，必须 ≤ alpha |
| `ramp_frames` | INT | 3 | 末尾线性降到 alpha_end 的帧数 |
| `seed` | INT | 0 | 色块图案种子，可复现 |

输出仍是 `IMAGE`。色块网格固定 `36×64`（576×1024 时为 16×16 像素块），与原脚本一致。

### My SelfLift Progressive Sampler (MiniMax H3)

基于 `facok/comfyui-SelfLift` 的 MiniMax H3 渐进分辨率采样节点，复制自
`3534184`。节点 ID 为 `MySelfLiftH3Sampler`，不会覆盖原插件的
`SelfLiftH3Sampler`；已有工作流需要手动替换为本节点。

本地改动：3D latent upscaler 完成分辨率跃迁后，立即通过 ComfyUI 模型管理器
从活跃 GPU 模型集合卸载。正常 Dynamic VRAM 模式下，
`minimax_h3_latent_upscaler_3d_fp32.pth` 的约 1.29 GiB 权重会回到 CPU，
再进入 MiniMax H3 高分辨率去噪；发生放大异常时也执行清理。放大结果和采样数学
不变，后续任务再次使用该放大器时会重新载入权重。`--highvram` / `--gpu-only`
模式的 offload device 仍可能是 GPU，此时不会获得同等显存回收。

### Patch Sol-Attn (MiniMax)

第三方节点，代码在 `my_nodes/vendor/`。在 MiniMax-H3 上安装 block-sparse
attention（Sol-Attn），长序列（≥ ~12k tokens）提速；不满足条件的调用
（短序列 / 非 bf16 / 非 128 head_dim / cross-attention 等）自动回落到
ModelAttentionBackend 已有的 attention。

| 输入 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `model` | MODEL | — | H3 模型 |
| `tau` | FLOAT | 1.3 | 路由阈值，越高越稀疏（1.0≈16% blocks 精确，1.5≈7%，2.0≈2.7%） |
| `start_percent` | FLOAT | 0.2 | 该采样阶段之前保持 dense |
| `end_percent` | FLOAT | 0.9 | 该采样阶段之后保持 dense |
| `min_tokens` | INT | 12288 | 序列短于此值不走稀疏 |
| `sink_conditioning` | COMBO | `exact_kv_and_rows` | 条件行保持精确（exact_kv / exact_kv_and_rows / off） |
| `morton` | BOOLEAN | false | 视频 token 重排成 Morton 顺序 |
| `morton_curve` | COMBO | `2d_frame` | Morton 曲线（3d / 2d_frame） |
| `centroid_tail` | BOOLEAN | true | 每个 query block 用质心做 pooled branch（关闭做质量 A/B） |
| `routed_cap_percent` | INT | 0 | 路由块列表上限百分比，0=不限 |
| `reuse_qkv_memory` | BOOLEAN | false | 复用 fused qkv buffer 写输出，省约 1.2 GB @80k tokens |
| `verbose` | BOOLEAN | false | 详细日志 |
| `tau_profile` | STRING | 空 | 逐 block tau，如 `39-42=0.9` |
| `dense_blocks` | STRING | 空 | 保持 dense 的 block，如 `0-2,-1` |

### TE-Speed VOSR2（Linux 原生）

纯 Python/Torch 的 Linux 后端，节点 ID、端口类型、widget 顺序与
`TE-Speed-VOSR2 1.0` Windows 插件兼容，不加载 `.pyd`，也不会联网下载模型。
提供四个节点：

- `TESpeedVOSR2Loader`
- `TESpeedVOSR2Settings`
- `TESpeedVOSR2Image`
- `TESpeedVOSR2Video`

模型 bundle 放在 `<ComfyUI models>/vosr2/VOSR2/`：

```text
VOSR2/
├── args.json
├── ema_model.safetensors          # 也接受 checkpoints/ 或 clean_weights/
├── dinov2_vitl14.safetensors      # Hugging Face/Meta 键格式均可
└── Qwen-Image-vae-2d/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

推理会自动采用 ComfyUI 已启用的 SageAttention，缺失时回退到 PyTorch
SDPA。`memory_policy=staged` 每阶段通过 ComfyUI `ModelPatcher` 在 CPU/GPU
间换载，适合 24 GiB 显存；`resident` 尝试让 DiT、DINO 和 VAE 常驻，
VAE fp32 解码空间不足时仍会主动释放两个 Transformer。VAE 解码始终为
fp32；`vae_encode_amp` 只影响编码。

RTX 3090 建议先用默认 `speed`、`tile_size=512`、`vae_tile_size=1024`；
显存不足时将 memory policy 设为 `staged`，并降低 image/frame/DINO
batch。视频节点可选的 `temporal_cache` 只复用变化低于阈值的相邻帧
DINO 特征，并由 `cache_refresh` 强制周期刷新。

移植代码及上游许可证说明见
[`my_nodes/core/vosr2/THIRD_PARTY_NOTICES.md`](my_nodes/core/vosr2/THIRD_PARTY_NOTICES.md)。

### My Video Enhance

`IMAGE → IMAGE`，全部阶段关闭时输入对象原样返回（pass-through 兼容）。三个开关独立；两个阶段同时开启时先后由 `stage_order` 决定，默认 `dlss_then_vfi` 就是旧的固定顺序。默认链路：DLSS feature 1（DLAA / 超分）→ 可选 feature 18 神经渲染（同一个 Wine worker）→ worker 退出后，可选离线 GIMM-VFI 2x；`vfi_then_dlss` 则先插帧，再增强插出来的帧。关掉的阶段不会被调用；打开但缺文件时直接报路径，没有静默回退。

| 输入 | 默认 | 含义 |
|---|---|---|
| `enable_super_resolution` | false | feature 1。`1.0 DLAA (native)` 是原生分辨率抗锯齿，不是放大 |
| `spatial_mode` | `2.0x` | `1.0 DLAA (native)` / `1.5x` / `2.0x` / `3.0x` |
| `enable_neural_rendering` | false | feature 18，实验性。`light/standard/portrait/detail/custom` 是本节点的 UX 映射，不是 NVIDIA 官方预设 |
| `nr_profile` / `nr_intensity` | `standard` / `1.0` | 上面的 UX profile 与强度（0.0–2.0）。内置 profile 的模型字段固定在表里，只由 `nr_intensity` 换强度；`custom` 反过来读取下面 7 个模型字段 |
| `style` / `preset` | `Cinematic` / `Default` | 高级项，仅 `nr_profile=custom` 读取。`style`：`Default` / `Natural` / `Cinematic`；`preset`：`Default` / `Preset 1` / `Preset 2` / `Preset 3`（映射为 DLSSNR 的 0/1/2 与 0/1/2/3） |
| `local_structure` / `local_tone` | 1.0 / 1.0 | 高级项，仅 `custom`。0–2，对应 DLSSNR 的局部结构强度与局部影调强度 |
| `skin` | -1.0 | 高级项，仅 `custom`。-1 保持模型默认（不下发该参数），0–2 是显式强度 |
| `ui_correction` / `auto_mask` | false / false | 高级项，仅 `custom`。两者都进入 DNR3 头；当前原生桥还从显式子进程环境读取同值的 `ui_correction`，`auto_mask` 只需头字段 |
| `detail` / `color` | 1.0 / 1.0 | 高级项，**只要跑了神经渲染就生效，与 profile 无关**：`detail` 是与 NR 前帧的整体混合强度（0–2），`color` 是从 NR 前色调过渡到 NR 结果色调的比例（0–1）。1.0 / 1.0 等于原始模型输出 |
| `sr_preset` | `Default` | 高级项，超分阶段生效。DLSS 超分模型档位：`Default` / `E` / `F` / `J` / `K` / `L` / `M`，对应 NGX 的 0/5/6/10/11/12/13；`Default` 交给运行时自己选 |
| `gpu_index` | 0 | 高级项。DLSS worker 使用的 GPU 序号（0–15），0 是第一块可见 GPU |
| `enable_frame_interpolation` | false | 离线 GIMM-VFI-R，固定 2 倍。不是 DLSS Frame Generation |
| `vfi_precision` / `vfi_ds_factor` | `fp32` / `1.0` | 高级项。权重必须已在磁盘上，节点不会下载 |
| `motion` / `scene_cut_threshold` | `optical_flow` / `0.2` | 高级项。DIS 估计当前帧到前一帧的反向像素光流；切镜时重置时序历史 |
| `channel_order` | `auto` | 高级项。DNR3 回读纹理的通道顺序：`auto` 用第一帧对比源帧判定一次，然后整段沿用；异常时可强制 `RGBA` / `BGRA` |
| `runtime_dir` / `wine_prefix` | 空 | 高级项。空时依次回退环境变量与默认路径，见下面的配置指南 |
| `worker_timeout` | 600.0 | 高级项。DLSS worker 单次读写等待的超时秒数（1–86400），不是整段视频总时限 |
| `stage_order` | `dlss_then_vfi` | 高级项。两个阶段同时开启时的顺序：`dlss_then_vfi` 先 DLSS 再对结果插帧，`vfi_then_dlss` 先插帧再增强。只有一个阶段时该值不改变行为 |

N 帧插值后是 `2*N-1` 帧。`fps_multiplier` 只在插值真正跑过时为 2，编码时用输入 FPS 的 2 倍。单帧和关闭插值都是 1。

`nr_profile=custom` 是本节点对 feature 18 的完整手动控制：`style` / `preset` / `local_structure` / `local_tone` / `skin` / `auto_mask` / `ui_correction` 映射到 DLSSNR 的模型参数（`global_tone` 不开放，保持模型默认）。模型字段与强度随 DNR3 头下发；`ui_correction` 还会和 `detail` / `color` / `sr_preset` / `gpu_index` / 请求的 `channel_order` 一起通过子进程环境显式下发——每次启动都写入这 6 个变量，所以父进程里的同名变量不会改变本次运行，同一个 worker 的整段视频也不会中途换挡。所有高级项即使对应阶段关闭也会校验，取值非法直接报错，没有静默回退；完全不碰这些高级项时行为与旧版本一致（`detail=1.0`、`color=1.0`、UI 校正关闭、SR preset `Default`、GPU 0）。

#### 阶段顺序、磁盘中间帧与内存

- 两个阶段同时开启时，第一阶段的结果完整写进 Comfy 临时目录下的 float32 `np.memmap` 中间文件；第一阶段的模型/子进程完全拆除后第二阶段才开始，所以 GIMM 与 Wine/DLSS worker 不会同时在显存里。文件按固定大小分块映射，默认每块最多 256MiB（至少一整帧），切换时释放旧映射，避免整文件映射的 RSS 随片长增长。
- 中间文件需要的空闲空间是 `中间帧数 × H × W × 3 × 4` 字节：`dlss_then_vfi` 的中间帧是超分后的尺寸，`vfi_then_dlss` 的是 `2N-1` 帧的原始尺寸。空间不足在开跑前就会报出所需/可用字节数。临时中间文件在成功、报错和 Comfy 取消时都会删掉。
- 返回的 `IMAGE` 必须是一整块常驻内存的 float32 批次。节点在开跑前按最终 `N×H×W×3×4` 精确预检，超过**当前可用 RAM 的 80%** 直接拒绝（错误里给出所需/可用字节数），并建议改用 `MyVideoEnhanceStream`——它把帧留在磁盘上。

#### DLSS 运行时与 Wine 依赖配置指南

本节点通过独立的 Wine 隔离子进程调用 NVIDIA Windows 原生 NGX 运行时。本仓库不分发闭源专有文件，需要配置以下运行环境：

##### 1. NVIDIA 运行时 DLL 目录（`<ComfyUI>/models/dlss5/`）
将以下 DLL 放置在 `<ComfyUI>/models/dlss5/`（或在节点的 `runtime_dir` 填写绝对路径）：
- `_nvngx.dll` & `nvngx.dll`：NVIDIA 驱动 NGX 核心。Linux 系统安装官方驱动后可直接从 `/usr/lib/nvidia/wine/` 获取：
  ```bash
  mkdir -p models/dlss5
  cp /usr/lib/nvidia/wine/_nvngx.dll /usr/lib/nvidia/wine/nvngx.dll models/dlss5/
  ```
- `nvngx_dlss.dll`：DLSS 超分辨率（Super Resolution / DLAA）官方运行库（可从 NVIDIA 官方 DLSS SDK 或 Windows 游戏安装目录获取）。
- `nvngx_dlssnr.dll`（可选）：DLSS 神经渲染（Neural Rendering）实验性库。

##### 2. Wine 前缀与 Direct3D 12 运行库（`~/.wine`）
默认使用 `~/.wine`（或在节点的 `wine_prefix` 填写自定义前缀）。子进程通过 D3D12 调用 GPU，必须将以下 64 位 DLL 部署到 `drive_c/windows/system32/`：
- `d3d12.dll` & `d3d12core.dll`：来自 [vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton)（注意：两者必须同时存在，现代 vkd3d-proton 依赖 `d3d12core.dll` 提供 Agility 核心接口）。
- `nvapi64.dll`：来自 [dxvk-nvapi](https://github.com/jp7677/dxvk-nvapi)。
- `dxgi.dll`：来自 [dxvk](https://github.com/doitsujin/dxvk)。

*若系统安装了 Steam Proton（如 Proton Experimental），可一键提取现成的 64 位组件：*
```bash
wineboot -u
PFX_SYS32="$HOME/.wine/drive_c/windows/system32"
PROTON_FILES="$HOME/.local/share/Steam/steamapps/common/Proton - Experimental/files"

cp "$PROTON_FILES/lib/wine/vkd3d-proton/x86_64-windows/d3d12.dll" "$PFX_SYS32/"
cp "$PROTON_FILES/lib/wine/vkd3d-proton/x86_64-windows/d3d12core.dll" "$PFX_SYS32/"
cp "$PROTON_FILES/lib/wine/nvapi/x86_64-windows/nvapi64.dll" "$PFX_SYS32/"
cp "$PROTON_FILES/lib/wine/dxvk/x86_64-windows/dxgi.dll" "$PFX_SYS32/"
```

##### 3. 显示环境与无头服务器（Headless Server / Docker / 云 GPU）支持

Wine 的 Direct3D 12 驱动在初始化交换链与离屏渲染上下文时，需要一个 X11 Display 协议端点。纯后台计算依然 100% 直通调用物理 NVIDIA 显卡，但需根据运行环境提供相应的 `DISPLAY`：

- **桌面开发机 / 物理机（已有 Xorg 或桌面环境，但通过 systemd 后台服务启动）**：
  直接在启动脚本（如 `run-comfyui`）中导出当前桌面的 Display：
  ```bash
  export DISPLAY="${DISPLAY:-:0}"
  ```

- **纯无头服务器（Linux Server / 云端 GPU 如 AutoDL、RunPod / Docker 容器）**：
  纯命令行系统没有物理显示器和桌面环境，推荐使用轻量级虚拟帧缓冲 **`Xvfb`**（不消耗真实显示资源，内存中模拟端点）：
  1. 安装 `Xvfb`：
     - Ubuntu / Debian: `apt-get update && apt-get install -y xvfb`
     - Arch Linux: `pacman -S xorg-server-xvfb`
  2. 后台启动虚拟显示服务：
     ```bash
     Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
     export DISPLAY=:99
     ```

- **推荐自适应启动脚本写法（桌面机与无头 Server 通用）**：
  ```bash
  if [ -z "${DISPLAY:-}" ]; then
    if ! pgrep -x Xorg >/dev/null 2>&1 && command -v Xvfb >/dev/null 2>&1; then
      # 纯无头环境：自动拉起 Xvfb 虚拟显示
      Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
      export DISPLAY=:99
    else
      # 本地桌面/服务环境：默认连接物理 :0
      export DISPLAY=:0
    fi
  fi
  ```

##### 4. 硬件与驱动兼容性注意事项
- **DLSS-SR（超分辨率 / DLAA 1.0x / 1.5x / 2.0x / 3.0x）**：
  - 支持 RTX 20/30/40 全系列 GPU。
  - 在 Linux + RTX 3090 + NVIDIA 驱动（已验证 610.57+）下已全面测试通过，运行稳定流畅。
- **DLSS-NR（神经渲染 Feature 18）**：
  - 属于 Linux/Wine 上的非官方实验性功能；RTX 3090 不在 NVIDIA 官方 DLSS 5 支持范围内。此前写的“Linux 驱动 ≥ 616.56”不成立：616.56 是 Windows 驱动版本，不能与 Linux 驱动号直接比较。当前尚未验证 Linux 615.71.09 上 Feature 18 可稳定完成推理，可能遇到 GPU fence 同步超时；生产使用请保持 `enable_neural_rendering: false`，仅在驱动内核模块与用户态版本一致后进行隔离的小尺寸探测。参见 [NVIDIA Linux 615.71.09 驱动](https://www.nvidia.com/en-us/drivers/details/278450/)与 [DLSS 5 官方支持范围](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/)。
- **插帧（VFI）**：
  - RTX 3090 硬件不支持 DLSS 3 Frame Generation（仅 RTX 40+ 支持）。本节点内置了基于 S-Lab GIMM-VFI 的纯离线高质量光流补帧，在独立阶段执行，不受 DLSS 硬件代际限制；默认顺序 `dlss_then_vfi` 下它跑在 Wine worker 退出之后，`vfi_then_dlss` 下跑在 worker 启动之前。
  - 权重位于 `models/interpolation/gimm-vfi/gimmvfi_r_arb_lpips_fp32.safetensors` 和 `raft-things_fp32.safetensors`。

### My Video Enhance Stream

原生 `VIDEO → VIDEO, INT, STRING`（`video` / `interpolation_multiplier` / `status`）。它把与 `My Video Enhance` 完全相同的阶段一次一帧地跑完，长片段不需要把整段帧塞进 RAM。DLSS 运行时、Wine 前缀与驱动要求同上一节。推荐接线：

```text
LoadVideo
  → My Video Enhance Stream      # VIDEO + 倍数 + 状态串
    → SaveVideo                  # 推荐 format=mkv、codec=auto，避免重复编码
```

可导入示例：[video_enhance_stream.json](docs/video_enhance_stream.json)。默认先用 fp16 插帧、再 DLSS 2x，NR 关闭；加载后在 `LoadVideo` 选择自己的输入文件。

增强开关与高级项和上一节相同（`enable_super_resolution` / `spatial_mode` / `enable_neural_rendering` / `nr_profile` / `nr_intensity` / `style` / `preset` / `local_structure` / `local_tone` / `skin` / `detail` / `color` / `ui_correction` / `auto_mask` / `sr_preset` / `gpu_index` / `enable_frame_interpolation` / `vfi_precision` / `vfi_ds_factor` / `motion` / `scene_cut_threshold` / `channel_order` / `runtime_dir` / `wine_prefix` / `worker_timeout`，含义、默认值与生效范围见上一节：模型字段只在 `nr_profile=custom` 下生效，`detail` / `color` 在启用神经渲染时始终生效）。流式节点新增或默认值不同的输入：

| 输入 | 默认 | 含义 |
|---|---|---|
| `video` | — | 本地的、可 seek 的、文件型 VIDEO |
| `output_codec` | `libx264` | 编码器。`libx264` 是 CPU 默认；`h264_nvenc` 走 NVENC，需要受支持的 GPU |
| `quality` | 18 | 0–51，越小越好也越大：libx264 是 CRF，h264_nvenc 是 CQ |
| `stage_order` | `vfi_then_dlss` | 高级项。**默认与 IMAGE 节点相反**：两阶段同时开启时先插帧，让光流落在原始分辨率上，比先超分再对放大帧做光流更省显存 |

#### v1 接受的输入

只接受本地、可 seek 的**文件型** VIDEO，必须是整段未裁剪、CFR、8-bit SDR。以下情况直接报错，不会静默降级：

- 可变帧率（VFR）或时间戳与声明帧率不符；URL / 管道 / 实时源。
- 非文件型 VIDEO：内存型或组件型，以及没有原生 VIDEO API 的输入。
- 带 active trim window 的 VIDEO——请先将裁剪结果保存为独立文件，再重新 `LoadVideo`；仅连接 `Trim Video` 不会物化裁剪结果。
- HDR（PQ / HLG / BT.2020）、任何非 8-bit 输入（包括低位深与 10/12-bit）及浮点像素格式。
- 带旋转、翻转等非单位显示矩阵或非方形像素（SAR 不为 1:1）的文件；需先归一化，错误信息中提供操作指引。

#### 音频与输出

- 源文件的第一条音轨原样 stream copy 进输出，并保留它相对首个解码视频帧的 PTS 起始偏移；不合成或重编码音频，源没有音频就没有音轨。字幕、metadata、其余音轨不做保留承诺。
- 输出固定 8-bit `yuv420p` Matroska，写在 Comfy 临时目录（`my_nodes_enhance_stream_*_<uuid>.mkv`），包装成 `VideoFromFile` 返回；要长期保存由下游 `SaveVideo` 决定落盘位置。
- 输出宽高必须是偶数——`yuv420p` 的要求，不满足时编码器直接报出实际宽高。DLAA（1.0x）保持源尺寸，所以奇数尺寸的源在这一档会直接失败；超分档由节点向上取偶。
- `status` 里带阶段、帧数、`output_fps`、`codec`、`quality` 以及音频处理结果。

#### 帧数与 FPS

插帧固定 2 倍：`N` 帧进、`2N-1` 帧出。输出 FPS 只在插帧真的有源帧对（源至少 2 帧）时才是源 FPS 的 2 倍；单帧片段保持源 FPS，`interpolation_multiplier` 也返回 1。两个 `stage_order` 都保证 GIMM 与 Wine/DLSS 不同时驻留。

#### 资源与临时文件

- 单阶段只保留少量帧缓冲；双阶段额外保留一个映射块，默认最多 256MiB（至少一帧）。固定分辨率下，帧缓冲与映射的 RSS 上界不随片长增长；模型、编解码器缓冲以及系统可回收文件缓存另计。
- 中间 float32 暂存同样是 `中间帧数 × H × W × 3 × 4` 字节，写在 Comfy 临时目录下，建议使用磁盘而非 tmpfs；最终编码及音频复用文件另外占空间。节点预检中间文件所需空间，不预估压缩后文件大小，也不能阻止其他进程运行中占满磁盘。
- 中间帧 memmap、只写视频的中间文件、失败/取消时的半成品输出在成功、报错和 Comfy 取消时都会清掉；成功后的最终临时文件保留，因为返回的 VIDEO 仍指向它。
- 源文件解码两遍：一次 ffprobe 的 O(1) 内存 CFR / 帧数扫描，一次真正的逐帧处理。
- 全部阶段关闭时是纯 pass-through：不探测、不解码、不编码，输入 VIDEO 原样返回。

#### 实测记录

- RTX 3090 上两个 `stage_order` 均通过真实跨分块验证：3 帧 → 5 帧、128×128 → 256×256、3fps → 6fps，AAC 音轨保留；`h264_nvenc` 也实测通过。
- 实际 720p 素材：241 帧 1280×720/24fps，经 fp16 VFI → DLSS 2x → libx264 quality 18，得到 481 帧 2560×1440/48fps，音轨保留，耗时约 515 秒。主进程阶段采样 RSS 为 2656–2970MiB，编解码器/Wine 子进程另计；进入 DLSS 后及运行结束时，PyTorch 已分配 CUDA 显存为 0。
- 中间帧存储的独立 RSS 回归覆盖 20,000 帧、245.8MB 原始文件：1MiB 测试分块下额外峰值 RSS 约 1MiB，而非整文件大小。以上不等同于小时级长视频、所有容器/音轨时间轴的生产验证。

### My DLSS Runtime Probe

用同一条 DNR3 管线跑一帧 32×32 的确定性 RGB，只返回状态字符串。至少打开 DLAA/超分或神经渲染。缺 DLL 或缺 Wine 前缀文件时，错误里会写出缺失路径。

## 加新节点

1. 无 Comfy 依赖的算法放 `my_nodes/core/<name>.py`。
2. 节点类放 `my_nodes/nodes/<name>.py`，同时提供经典 `INPUT_TYPES`（兼容旧加载器）。
3. 在 `my_nodes/registry.py` 的 `NODE_CLASSES` 里登记。
4. 在 `tests/` 补单元测试。

```text
my-comfyui-nodes/
├── __init__.py                 # Comfy 入口
├── my_nodes/
│   ├── registry.py             # 唯一登记处
│   ├── core/                   # 纯函数，可单测
│   ├── nodes/                  # 一个文件一个节点
│   └── vendor/                 # 第三方节点，字节级保留原文件
└── tests/
```

## 第三方 vendor 节点

`my_nodes/vendor/sol_attn_minimax_v2.py` 原样转存自
[t8star/Sol-Attn-v2-wheels](https://huggingface.co/t8star/Sol-Attn-v2-wheels)
（节点 `Patch Sol-Attn (MiniMax)`）。文件不做任何修改，更新时直接重新下载覆盖；
它依赖 comfy-kitchen 的 `sol_attn` 内核（Kijai `sol_attn` 分支），缺失时 registry
只跳过注册并在日志告警，不影响其余节点。

背景、协作原理、编译与回滚记录见 [`docs/SOL_ATTN.md`](docs/SOL_ATTN.md)。

## 测试

```bash
python3 -m unittest discover -s tests -v
```
