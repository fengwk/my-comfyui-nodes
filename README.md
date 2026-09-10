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
