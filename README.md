# my-comfyui-nodes

个人 ComfyUI 自定义节点仓。算法放在 `my_nodes/core/`，节点接线放在 `my_nodes/nodes/`，注册表集中在 `my_nodes/registry.py`。

## 安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/fengwk/my-comfyui-nodes.git
# 或: ln -s /path/to/my-comfyui-nodes /path/to/ComfyUI/custom_nodes/my-comfyui-nodes
```

依赖：`numpy`、`Pillow`（Comfy 环境一般已有）。

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
│   └── nodes/                  # 一个文件一个节点
└── tests/
```

## 测试

```bash
python3 -m unittest discover -s tests -v
```
