# my-comfyui-nodes

个人 ComfyUI 自定义节点仓。算法放在 `my_nodes/core/`，节点接线放在 `my_nodes/nodes/`，注册表集中在 `my_nodes/registry.py`。

## 安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fengwk/my-comfyui-nodes.git
# 或: ln -s /path/to/my-comfyui-nodes /path/to/ComfyUI/custom_nodes/my-comfyui-nodes
```

依赖：`numpy`、`Pillow`（Comfy 环境一般已有）。

可选：`vendor/sol_attn_minimax_v2.py` 需要 comfy-kitchen 的 `sol_attn` 内核
（Kijai `sol_attn` 分支构建，官方 PyPI 包不含）。缺失时该节点不注册，其余节点不受影响。

## 输入约定：用 IMAGE 批次，不要图片列表

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

### My Video Enhance

三个开关独立，顺序固定：DLSS feature 1（DLAA / 超分）→ 可选 feature 18 神经渲染（同一个 Wine worker）→ worker 退出后，可选离线 GIMM-VFI 2x。关掉的阶段不会被调用；打开但缺文件时直接报路径，没有静默回退。

| 输入 | 默认 | 含义 |
|---|---|---|
| `enable_super_resolution` | false | feature 1。`1.0 DLAA (native)` 是原生分辨率抗锯齿，不是放大 |
| `spatial_mode` | `2.0x` | `1.0 DLAA (native)` / `1.5x` / `2.0x` / `3.0x` |
| `enable_neural_rendering` | false | feature 18，实验性。`light/standard/portrait/detail` 是本节点的 UX 映射，不是 NVIDIA 官方预设 |
| `enable_frame_interpolation` | false | 离线 GIMM-VFI-R，固定 2 倍。不是 DLSS Frame Generation |
| `vfi_precision` / `vfi_ds_factor` | `fp32` / `1.0` | 高级项。权重必须已在磁盘上，节点不会下载 |
| `motion` / `scene_cut_threshold` | `optical_flow` / `0.2` | 高级项。DIS 估计当前帧到前一帧的反向像素光流；切镜时重置时序历史 |

N 帧插值后是 `2*N-1` 帧。`fps_multiplier` 只在插值真正跑过时为 2，编码时用输入 FPS 的 2 倍。单帧和关闭插值都是 1。

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
  - 属于实验性功能，底层调度要求 NVIDIA 驱动版本 ≥ 616.56。在低于 616.56 的驱动上执行可能会因 GPU fence 同步挂起超时。若当前系统驱动未满足要求，请保持 `enable_neural_rendering: false`。
- **插帧（VFI）**：
  - RTX 3090 硬件不支持 DLSS 3 Frame Generation（仅 RTX 40+ 支持）。本节点内置了基于 S-Lab GIMM-VFI 的纯离线高质量光流补帧，在 Wine 进程退出后独立执行，不受 DLSS 硬件代际限制。
  - 权重位于 `models/interpolation/gimm-vfi/gimmvfi_r_arb_lpips_fp32.safetensors` 和 `raft-things_fp32.safetensors`。

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
