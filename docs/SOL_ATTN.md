# Sol-Attn v2 接入说明

2026-08 在 `/home/fengwk/prog/ComfyUI`（Linux + RTX 3090）接入 Sol-Attn v2 的记录。
目的：未来回顾时无需重新调研。

## TL;DR

- ComfyUI 的 attention 可插拔：`ModelAttentionBackend` 把 `comfy kitchen attention`
  注册为模型的 `optimized_attention_override`；`Patch Sol-Attn (MiniMax)` 在其上再套
  一层 override——符合条件的 self-attention 走 `comfy_kitchen.sol_attn()`，其余**链式
  回落**到前一层（即 `comfy kitchen attention`）。
- 官方 PyPI `comfy-kitchen==0.2.31` **不含** `sol_attn`；带内核的版本在 Kijai
  `sol_attn` 分支（PR #117 未合并），本机装的是**从该分支源码本地编译的 wheel**。

## 仓库关系

| 仓库 | 角色 |
|---|---|
| [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | 官方内核库，PyPI 0.2.31 无 sol_attn |
| [PR #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117) | sol_attn 上游 PR，Draft 未合并 |
| [kijai/comfy-kitchen `sol_attn` 分支](https://github.com/kijai/comfy-kitchen/tree/sol_attn) | 内核源码（接入时 head = `26d7d6c`） |
| [t8star/Sol-Attn-v2-wheels](https://huggingface.co/t8star/Sol-Attn-v2-wheels) | 节点文件 `sol_attn_minimax_v2.py` + Windows wheel 转存 |

HF 节点是"接线层"，不带 CUDA 代码，运行时调用 `comfy_kitchen.sol_attn(...)`。

## 协作原理

### 一次 attention 调用的完整路径

```text
attention 调用
  └─ Sol-Attn override（Patch Sol-Attn 安装）
       ├─ 符合条件 → comfy_kitchen.sol_attn()        # block-sparse INT8 CUDA
       └─ 不符合   → previous override（ModelAttentionBackend 安装）
                       └─ comfy_kitchen.int8_attention()   # dense INT8 CUDA
```

### 谁装了什么

- `ModelAttentionBackend`（`comfy_extras/nodes_model_advanced.py`，`ModelAttentionBackend`）
  把 `comfy_kitchen_int8` 包成 override 写入
  `transformer_options["optimized_attention_override"]`。
- Sol-Attn 节点的 `_apply_patch` 读取这个**已有 override 存为 `previous`**，再写入自己
  的 `make_override(previous=...)`。每个 attention 调用按序判断：
  有 mask / 非自注意力 / 非 bf16 / head_dim≠128 / 序列 < `min_tokens` / sigma 窗口外 /
  dense_blocks 内 → 不适用，交给 `previous`；否则调用 `comfy_kitchen.sol_attn()`。
- ComfyUI 的 override 入口在 `comfy/ldm/modules/attention.py` 的 `wrap_attn`：每次
  attention 前查 `optimized_attention_override`，有就用它接管调用。

### 节点顺序

```text
MODEL → ModelAttentionBackend (comfy kitchen attention)
      → Patch Sol-Attn (MiniMax)
      → Sampler
```

不能反：后置节点会整体覆盖 `optimized_attention_override`，Sol-Attn 会被丢掉。

### 与 KJNodes 的桥接

KJNodes 的 `MiniMaxLowVRAMAttention` 写入 `transformer_options["sol_take_forward"]` 并
给 forward 打 `_uses_optimized_attention = True`。Sol-Attn 的 `_install_compose_hooks`
据此区分：标记过的 patch 内部走 `optimized_attention`，override 直接组合；否则用
`_compose_module_patch` 包一层 gate，符合条件的调用走 `sol_take_forward`。三者协同即：
Sol-Attn 优先，低显存/Sage patch 保持其自身行为。

## 编译

> 结论：**本地编译，只编 CUDA，架构 80/86/89（native cubin）**。

原因：PyPI 0.2.31 无 sol_attn；GHA artifact 需 GitHub 登录；HF 只有 Windows wheel。

```bash
git clone --depth 1 --branch sol_attn \
  https://github.com/kijai/comfy-kitchen.git /tmp/ck_sol_attn
cd /tmp/ck_sol_attn
git submodule update --init --recursive --depth 1   # cutlass + flash-attention

python3 -m venv /tmp/ckbuildvenv
/tmp/ckbuildvenv/bin/pip install nanobind cmake ninja

cd /tmp/ck_sol_attn
CUDA_HOME=/opt/cuda \
COMFY_CUDA_ARCHS="80-real;86-real;89-real" \
/tmp/ckbuildvenv/bin/pip wheel . --no-deps --no-build-isolation -w /tmp/ck_wheelhouse
```

产物：`comfy_kitchen-0.2.31-cp312-abi3-linux_x86_64.whl`（19 MB）。

要点：

- **为什么必须指定 arch**：官方默认列表含 80/89/90/100/120，**没有 86**（3090）。默认
  构建只能靠 PTX JIT 跑 3090，不稳定。
- **`-real` 不含 PTX**：wheel 只跑 SM80/86/89（3090/4090/A100 等），不能跨架构 JIT。
- **跳过了 HIP**：AMD 不可用，省 ~200 MB；wheel 无 HIP `.so` 时导入正常（仅显示
  `hip: unavailable`）。

验证记录（2026-08-22）：3090 上 `sol_attn` 跑通，输出 bf16、与 dense SDPA 余弦约
0.97（INT8 近似符合预期）；ragged tail / sink / max_blocks 路径正常。

## 安装与切换

自编 wheel 有**两份持久副本**：

- 本仓库 `wheels/comfy_kitchen-0.2.31-cp312-abi3-linux_x86_64.whl`（随 git 分发）；
- `/home/fengwk/sol_attn_backup/`（本机备份目录，含 PyPI 原目录与切换脚本）。

```bash
bash /home/fengwk/sol_attn_backup/install_solattn.sh   # 装 sol_attn wheel
bash /home/fengwk/sol_attn_backup/restore.sh           # 回 PyPI 官方 0.2.31
```

节点文件 vendor 在 `my_nodes/vendor/sol_attn_minimax_v2.py`（HF 原始文件，字节级未改），
由 `my_nodes/registry.py` 惰性导入注册；缺 sol_attn 内核时该节点不注册，不影响其他节点。

装完后探针（应当全 True）：

```python
import comfy_kitchen as ck
hasattr(ck, "sol_attn")              # True
hasattr(ck, "int8_attention")        # True
ck.int8_attention_is_available()     # True
```

## 升级 ComfyUI 后恢复（重点）

升级方式默认是 `uv pip install -r requirements.txt`，这会把 comfy-kitchen 换回官方
PyPI 包（**sol_attn 丢失**），且本仓库 `uv.lock` 为空，任何 `uv sync` 都可能清掉整个
venv 的手工包。恢复步骤：

```bash
# 1. 验证 sol_attn 是否丢失
/home/fengwk/prog/ComfyUI/.venv/bin/python -c \
  "import comfy_kitchen as ck; print(hasattr(ck, 'sol_attn'))"

# 2. 若为 False，用仓库里或本机备份的 wheel 重装
bash /home/fengwk/sol_attn_backup/install_solattn.sh
# 或直接指定仓库 wheel：
/home/fengwk/prog/ComfyUI/.venv/bin/pip install --force-reinstall --no-deps \
  /home/fengwk/comfyui_data/custom_nodes/my-comfyui-nodes/wheels/comfy_kitchen-0.2.31-cp312-abi3-linux_x86_64.whl

# 3. 重启 ComfyUI，再跑一次探针确认
```

> 注意：自编 wheel 的版本号保持 0.2.31，与 `requirements.txt` 的 pin 一致，所以
> 正常 `uv pip install -r` 不会覆盖它；只有官方发布新版本号、`requirements.txt` 跟着
> 升级时才会被替换。届时按上面步骤恢复即可。

## 首次测试配置

```text
tau                1.3
start_percent      0.20
end_percent        0.90
min_tokens         12288
sink_conditioning  exact_kv_and_rows
morton             false
centroid_tail      true
routed_cap_percent 0
reuse_qkv_memory   false
verbose            true
```

终端看 `[sol_attn]` 前缀日志：`sparse` = 内核真正触发；`dense ...: <reason>` = 回落原因。
序列 ≥ `min_tokens` 才可能 sparse（短序列时 dense 通常更快，属正常）。

## 常见坑

- **同名版本**：官方 PyPI 与分支都叫 0.2.31，靠 `hasattr(ck, "sol_attn")` 区分。
- **节点重复注册**：同一节点文件不能放多个 custom_nodes 目录（ComfyUI 会报错）。现只
  保留 `my-comfyui-nodes` 一处。
- **重启生效**：替换 wheel / 新增节点后必须重启 ComfyUI。
- **uv.lock 形同虚设**：`uv run` 不会按 lock 重装，手工装的 wheel 不会被冲掉；但升级
  ComfyUI 依赖时可能被覆盖，需留意。
- **更新节点**：重新下载 HF 文件覆盖 `my_nodes/vendor/sol_attn_minimax_v2.py`，前提是
  类名 `SolAttnMiniMax` 与 `define_schema()` 接口不变。
- **更新内核**：Kijai 分支有新 commit 时，按上面编译命令重跑（换新目录即可），把新
  wheel 替换进 `sol_attn_backup/`。
