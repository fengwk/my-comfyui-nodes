# Sol-Attn v2 接入说明

本文档记录 2026-08-22 在 `/home/fengwk/prog/ComfyUI`（Linux + RTX 3090）上接入
Sol-Attn v2 的完整背景、仓库关系、协作原理、编译与回滚方式。写作目的：**未来自己
回顾时无需重新调研。**

## 0. TL;DR

- ComfyUI 的 attention 后端是**可插拔**的。`ModelAttentionBackend` 节点把
  `comfy kitchen attention`（comfy-kitchen 的 INT8 内核）注册成模型的
  `optimized_attention_override`。
- `Patch Sol-Attn (MiniMax)` 节点在模型上再套一层 override：符合条件的
  self-attention 调用走 `comfy_kitchen.sol_attn()`（block-sparse 内核），不符合的
  调用**链式回落到前一层 override**（即 `comfy kitchen attention`）。
- 关键约束：官方 PyPI 的 `comfy-kitchen==0.2.31` **不包含** `sol_attn`。真正带内核的
  版本在 Kijai 的 `sol_attn` 分支（未合并、未发版），本机用的是**从该分支源码本地编译
  的 wheel**。

## 1. 涉及的仓库与文件

| 仓库 / 位置 | 角色 |
|---|---|
| [`Comfy-Org/comfy-kitchen`](https://github.com/Comfy-Org/comfy-kitchen) | 官方内核库。PyPI 已发 0.2.31，但**无 sol_attn** |
| [`Comfy-Org/comfy-kitchen` PR #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117) | "Add int8 Sol-attention (CORE-391)"，Draft，未合并 |
| [`kijai/comfy-kitchen` `sol_attn` 分支](https://github.com/kijai/comfy-kitchen/tree/sol_attn) | 实际带 sol_attn 内核的源码。接入时 head = `26d7d6c`（GHA 官方构建也在此 commit） |
| [`t8star/Sol-Attn-v2-wheels`](https://huggingface.co/t8star/Sol-Attn-v2-wheels) | 节点文件 `sol_attn_minimax_v2.py` + Windows wheel 转存 |
| `/home/fengwk/prog/ComfyUI` | ComfyUI 运行目录（v0.33.1，含 venv） |
| `/home/fengwk/comfyui_data/custom_nodes/my-comfyui-nodes` | 个人节点仓，vendor 该节点 |
| `/home/fengwk/comfyui_data/custom_nodes/ComfyUI-KJNodes` | 第三方节点，预留 `sol_take_forward` 桥接字段 |

**关系**：HF 的节点文件是"接线层"，comfy-kitchen 分支是"内核层"。节点自己不带 CUDA
代码，它调用 `comfy_kitchen.sol_attn(...)`。

## 2. 协作原理：三层 override 链

### 2.1 ComfyUI 的 attention override 机制

ComfyUI 中每个模型的 attention 调用点最终都会经过
`comfy/ldm/modules/attention.py` 的 `wrap_attn` 装饰器（`attention.py:171-204`）。它
在调用真正 attention 前检查 `transformer_options["optimized_attention_override"]`：

```text
如果存在 optimized_attention_override
    → override(func, q, k, v, heads, ...)    # 接管，func 是原 attention
否则
    → func(q, k, v, heads, ...)               # 原路径
```

所以 override 的本质是：**一个函数，接管 attention 调用，内部可以自己决定算不算、
以及失败/不适用时把调用交回 `func`。**

### 2.2 ModelAttentionBackend 做了什么

`comfy_extras/nodes_model_advanced.py:352-382`：

- 读取用户选的 backend 字符串（`comfy kitchen attention` → `comfy_kitchen_int8`）；
- `get_attention_function(...)` 取到已注册的 INT8 内核函数；
- `model.set_model_optimized_attention(fn)`（`model_patcher.py:688-694`）把它包装成
  一个 override 放进 `transformer_options["optimized_attention_override"]`。

### 2.3 Patch Sol-Attn 做了什么

节点文件 `my_nodes/vendor/sol_attn_minimax_v2.py` 的 `_apply_patch`
（第 639-710 行）：

1. 读取模型上**已有的 override**（即上面那层），存为 `previous`；
2. 用 `make_override(previous=previous, ...)` 生成新 override，覆盖写回
   `transformer_options["optimized_attention_override"]`；
3. 对每个 attention 调用，新 override 的决策顺序（`make_override`，
   第 497-559 行）：
   - 有 mask → 不适用，走 `dense()`；
   - block 在 `dense_blocks` / 采样百分比在 `start/end` 窗口外 → 走 `dense()`；
   - 调用 `_run`（第 422-466 行），内部 `_ineligible`（第 400-419 行）逐项检查：
     - 没有 `comfy_kitchen.sol_attn` → 不适用；
     - 不在 CUDA / dtype 不是 bf16 / head_dim ≠ 128 → 不适用；
     - masked / cross-attention / q、k shape 不一致 → 不适用；
     - 序列长度 < `min_tokens`（默认 12288）→ 不适用；
   - 适用 → 调用 `comfy_kitchen.sol_attn(q, k, v, tau, scale, sink_blocks, ...)`；
   - 不适用 → `dense()`：**先交给 `previous`（即 comfy kitchen attention），没有才用
     原生 `func`**。

### 2.4 链式回落（fallback chain）

```text
attention 调用
  └─ Sol-Attn override（Patch Sol-Attn 安装）
       ├─ 符合条件 → comfy_kitchen.sol_attn()        # block-sparse INT8 CUDA
       └─ 不符合   → previous override（ModelAttentionBackend 安装）
                       └─ comfy_kitchen.int8_attention()   # dense INT8 CUDA
```

因此工作流中的节点顺序必须是：

```text
MODEL → ModelAttentionBackend (comfy kitchen attention)
      → Patch Sol-Attn (MiniMax)
      → Sampler
```

**不能反过来**：Sol-Attn 在前、ModelAttentionBackend 在后时，后置节点会直接覆盖
`optimized_attention_override`，Sol-Attn 被丢掉。

### 2.5 与 KJNodes 的桥接

KJNodes 的 `minimax_nodes.py:186-187` 会写入
`transformer_options["sol_take_forward"]`，并给函数打 `_uses_optimized_attention = True`
标记（第 137-138 行）。Sol-Attn 节点的 `_install_compose_hooks`（第 608-636 行）会检测
这类已打补丁的 forward：如果它标记了 `_uses_optimized_attention`，说明它内部走
`optimized_attention`，override 能直接组合；否则 Sol-Attn 会在 sampling 时用
`_compose_module_patch`（第 562-602 行）包一层 gate，让符合条件的调用走
`sol_take_forward`。这是为 KJNodes 的 `MiniMaxLowVRAMAttention` / Sage attention 补丁
预留的兼容层。

## 3. 为什么需要本地编译

1. **PyPI 0.2.31 没有 sol_attn**：实测 `hasattr(comfy_kitchen, "sol_attn") == False`，
   `list_backends()` 的 capabilities 列表里也没有它。
2. **官方 GHA 有 Linux 构建**：PR #117 的 CI 产物 `wheels-linux-cuda-hip` 带 sol_attn
   内核，但下载 Actions artifact **需要 GitHub 登录**，本机 `gh` 未登录。
3. **HF 只有 Windows wheel**：`t8star/Sol-Attn-v2-wheels` 只转存了 3 个 Windows zip，
   无 Linux 包。
4. 因此走第三条路：**从 Kijai `sol_attn` 分支源码本机编译**。

## 4. 编译方法（本机记录）

### 4.1 前提

- `/opt/cuda`（nvcc 13.3）、cmake ≥ 3.26、ninja、gcc/g++；
- 目标 GPU：RTX 3090 = compute capability **8.6**；
- 本机 torch = 2.12.0+cu130（Python 3.12.11）。

### 4.2 构建命令

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

产物：`/tmp/ck_wheelhouse/comfy_kitchen-0.2.31-cp312-abi3-linux_x86_64.whl`（19 MB）。

要点：

- `--no-build-isolation`：用独立构建 venv，不用 pip 临时拉依赖；
- `COMFY_CUDA_ARCHS`：只编本机需要的架构 cubin（官方默认列表
  `75-real;80-real;89;90a-real;100f;120f` 里 **没有 86**，所以默认构建在 3090 上靠
  PTX JIT，性能/稳定性没有保证）；
- `-real` 后缀：只产对应架构的 SASS cubin，**不内嵌 PTX**，因此这个 wheel 只能跑在
  SM80/86/89 上，不能跨架构 JIT；
- HIP 后端被跳过（`COMFY_KITCHEN_BUILD_HIP` 未设置）：NVIDIA 用不到，省 200 MB，
  且 AMD 机器不能用这个 wheel。

### 4.3 验证过的检查点

- wheel 内 `_C.abi3.so` 含 `sm_80/sm_86/sm_89.cubin` 及 `launch_sol_attn` 符号；
- RTX 3090 实测 `comfy_kitchen.sol_attn()` 正常执行，输出 bf16、cosine 与 dense
  SDPA 约 0.971（INT8 近似）；
- ragged tail（2033 tokens）、sink_blocks/sink_q、max_blocks 路径 finite 正常；
- 缺 HIP `.so` 时顶层 `import comfy_kitchen` 不崩，`list_backends()` 只显示
  `hip: unavailable`。

## 5. 安装记录

1. 备份原 PyPI 包：`/home/fengwk/sol_attn_backup/`（含 `comfy_kitchen_pypi/` 原目录 +
   `restore.sh`）；
2. 替换 venv 内包：
   ```bash
   /home/fengwk/prog/ComfyUI/.venv/bin/pip install --force-reinstall --no-deps \
     /tmp/ck_wheelhouse/comfy_kitchen-0.2.31-cp312-abi3-linux_x86_64.whl
   ```
3. 节点文件 vendor 进个人仓库：
   `my-comfyui-nodes/my_nodes/vendor/sol_attn_minimax_v2.py`（HF 原始文件，字节级未改）；
4. 注册：`my_nodes/registry.py` 惰性导入 + 条件注册。

安装后探针（应当全部 True）：

```python
import comfy_kitchen as ck
hasattr(ck, "sol_attn")              # True
hasattr(ck, "int8_attention")        # True
ck.int8_attention_is_available()     # True
```

## 6. 首次测试配置（交接文档建议）

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

观察终端日志前缀 `[sol_attn]`：`sparse` = 内核真正触发；`dense ...: <reason>` = 回落及
原因；只有序列 ≥ `min_tokens` 才可能 sparse（H3 短序列时 dense 通常更快）。

## 7. 回滚

```bash
bash /home/fengwk/sol_attn_backup/restore.sh
# 等价于：
# /home/fengwk/prog/ComfyUI/.venv/bin/pip install --force-reinstall --no-deps \
#   comfy-kitchen==0.2.31
```

## 8. 常见坑 / 备忘

- **同名版本**：官方 PyPI 与分支都叫 `comfy-kitchen 0.2.31`，不能靠版本号区分。判断
  依据是 `hasattr(comfy_kitchen, "sol_attn")`。
- **两个目录重复注册**：若把同一个节点文件放在多个 custom_nodes 目录，ComfyUI 会
  注册失败。现在只保留 `my-comfyui-nodes/my_nodes/vendor/` 一处。
- **ComfyUI 重启**：替换 wheel 或新增节点后必须重启 ComfyUI 进程才生效。
- **uv.lock 无效**：本仓库 `uv.lock` 几乎是空的（只有 virtual package），`uv run` 不会
  按 lock 重装依赖，所以手工装的 wheel 不会被冲掉；但反过来升级 ComfyUI 依赖时也
  可能被覆盖，需要留意。
- **更新 vendor 文件**：直接重新下载 HF 的 `sol_attn_minimax_v2.py` 覆盖
  `my_nodes/vendor/` 即可，前提是类名 `SolAttnMiniMax` 和 `define_schema()` 接口不变。

## 9. 时间线

| 时间 | 事件 |
|---|---|
| 2026-08-13 | PyPI 发布 comfy-kitchen 0.2.31（无 sol_attn）；ComfyUI v0.33.1 |
| 2026-08-21 | PR #117 最新 CI（run 32533223133）构建成功，产出 Linux artifact |
| 2026-08-22 | 本机：从分支 `26d7d6c` 本地编译 wheel，冒烟测试通过，替换安装 |
| 2026-08-23 | vendor 进 my-comfyui-nodes，提交 `c1cc0cc` 并推送 origin/main |
