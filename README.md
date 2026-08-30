# vllm_run — Qwen3.8-Flash-Next 多卡 vLLM 一键部署脚本

> 在 **8×NVIDIA A40（无 NVLink）** 服务器上，用 Docker + vLLM 一键部署
> `Qwen/Qwen3.8-Flash-Next-FP8`（125B MoE + 51B PLE n-gram + MTP）的实战脚本。
> 自动识别模型架构（dense / MoE / qwen4_exp），自动选择专用镜像与参数，无需手改。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 背景与调研结论](#2-背景与调研结论)
- [3. 快速开始](#3-快速开始)
- [4. 参数说明](#4-参数说明)
- [5. 使用示例](#5-使用示例)
- [6. 部署实战总结（4 卡落地配置）](#6-部署实战总结4-卡落地配置)
- [7. 实测效果](#7-实测效果)
- [8. 性能与时延](#8-性能与时延)
- [9. 踩坑记录（8 条，务必阅读）](#9-踩坑记录8-条务必阅读)
- [10. 故障排查](#10-故障排查)
- [11. 许可证](#11-许可证)
- [12. 免责声明](#12-免责声明)

---

## 1. 项目简介

| 项 | 值 |
|---|---|
| 脚本 | `vllm_run`（纯 bash，无 sudo 依赖，Docker 运行） |
| 实测模型 | `Qwen/Qwen3.8-Flash-Next-FP8`（官方 block-128 FP8，172.78 GiB，131 分片） |
| 实测硬件 | 8 × NVIDIA A40 48GB（Ampere sm_86，无 FP8 张量核）、**无 NVLink**（PCIe Gen4 x16）、503 GiB 内存、Ubuntu 22.04 |
| 引擎 | vLLM 专用镜像 `vllm/vllm-openai:qwen38-flash-next`（day-0 专用镜像，qwen4_exp 架构必需） |
| 落地配置 | 4 卡 TEP4（GPU 4-7）+ PLE CPU 卸载 + KV bf16 8.4GB + MTP3，256K 上下文 × 2 并发 |

**核心能力**

- 模型类型自动识别：读 `config.json` 判断 dense / MoE / qwen4，自动选择镜像、EP、MTP、PLE 卸载等参数；
- 多卡张量并行 + 专家并行（TP×DP 校验、GPU 空闲检测、手动 pin 卡）；
- Qwen3.8-Flash-Next 专属支持：51B PLE n-gram 表 CPU 卸载（`VLLM_PLE_CPU_OFFLOAD=1`）、GDN 线性注意力、QSA 稀疏注意力（BF16 KV 硬性要求）、MTP 投机解码；
- 工具调用（`--enable-auto-tool-choice` + `qwen3_xml` parser）+ 思维链解析（`qwen3`）；
- 无 sudo：日志、模型缓存全部位于用户 `$HOME` 下；`--restart always` 崩溃自动拉起；
- 无 InfiniBand 环境自动禁用 NCCL P2P/IB 走 TCP。

---

## 2. 背景与调研结论

> 以下结论来自对服务器硬件、模型生态、量化路线、并行策略的系统调研（详见本仓库关联文档）。

### 2.1 硬件关键结论（A40 服务器）

| 结论 | 依据 |
|---|---|
| **A40 没有 FP8 张量核**（sm_86），但权重可 **FP8 驻留** | TP1 实验实测：27B-FP8 加载后 `Model loading took 28.9 GiB`（每参数 1 字节，未反量化成 BF16），首次推理成功 |
| **无 NVLink** → 用 EP（专家并行）代替纯 TP | 纯 TP 每层 2 次 all-reduce，PCIe 上开销大；EP 的 all-to-all 流量小一个量级。且官方明确：**纯 TP 与 block-128 FP8 不兼容**（640 的 expert intermediate 无法按 128 量化块对齐） |
| **MTP 投机解码在卡内完成**，不增加跨卡流量 | 对无 NVLink 机器尤其友好 |
| 51B n-gram 表官方设计就是 CPU 驻留 + 异步预取 | `VLLM_PLE_CPU_OFFLOAD=1`，503GiB 内存绰绰有余 |
| 网络 | huggingface.co 直连被墙 → 用 `hf-mirror.com`；Docker Hub 直连被拒 → 用 DaoCloud 镜像源 |

### 2.2 模型结论（Qwen3.8-Flash-Next）

- 开源只有一种尺寸：**125B 主干（激活 6B）+ 51B n-gram 嵌入（PLE）+ 4B MTP** ≈ 180B 参数；
- **MoE**：48 层 × 512 专家（激活 10 routed + 1 shared），expert intermediate 640；
- **混合注意力**：每 4 层 1 层 QSA 全注意力 + 3 层 Gated DeltaNet 线性注意力；
- 原生上下文 **262,144**（YaRN factor 4 可到 1M）；多模态（image + video）；默认 thinking 模式；
- 架构 `qwen4_exp`（transformers 5.8.0.dev0）**必须用 day-0 专用镜像**，官方 nightly 镜像不认识该架构；
- 官方 FP8 block-128 量化（172.78 GiB），官方声明性能与原模型"几乎相同"。

### 2.3 量化选型结论（A40）

| 路线 | 结论 |
|---|---|
| 官方 FP8 block-128 | ✅ **选它**。A40 上权重 FP8 驻留（省一半显存 + 解码带宽减半），计算走 W8A16 反量化路径 |
| INT8 W8A16 | 理论更亲 A40（sm_86 原生 INT8 核），但 Flash-Next 无任何现成 INT8 checkpoint（社区全在 27B 上），自量化 170GB 成本高 → 仅作备胎 |
| NVFP4 | ❌ 仅 Blackwell（sm_100+）可跑；且文件与 FP8 几乎同大（专家 4bit，其余 BF16） |
| GGUF 4bit | ❌ llama.cpp 主线不支持 qwen4exp，且无 MTP 实现 |

**一句话**：官方 FP8 = 损失 ≈ 0 + 1 字节权重带宽，是 A40 上的最优点。

### 2.4 部署方案对比（最终选择 4C）

| 方案 | GPU | 并行 | n-gram 表 | 定位 |
|---|---|---|---|---|
| 8C TEP8 | 全机 8 卡 | TP8+EP | 可卸载 | 吞吐上限最高，需独占全机（本机有他人任务，未采用） |
| 6C TEP6 | 6 卡 | TP6+EP（专家 86/86/85/85/85/85 不均分，vLLM 原生支持余数） | GPU 驻留 | 折中方案 |
| **4C TEP4（落地）** | **GPU 4-7** | **TP4+EP4（512 专家均分 128/卡）** | **CPU 卸载（48GiB 进内存）** | **个人低并发：256K × 2，显存最小占用** |

---

## 3. 快速开始

```bash
# ① 拉取专用镜像（qwen4_exp 架构必需，非 nightly）
docker pull docker.m.daocloud.io/vllm/vllm-openai:qwen38-flash-next

# ② 下载模型（185GB，131 分片，走 hf-mirror）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.8-Flash-Next-FP8 \
  --local-dir /home/user/Project/vllm/models/Qwen3.8-Flash-Next-FP8

# ③ 脚本放到任意位置，赋执行权限
chmod +x vllm_run

# ④ 启动（默认 4 卡方案：GPU pin、PLE 卸载、TP4、8.4GB KV、端口 8000）
./vllm_run

# ⑤ 验证
curl http://127.0.0.1:8000/v1/models          # 应返回 Qwen-3.8-Flash-Next
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen-3.8-Flash-Next","messages":[{"role":"user","content":"你好"}]}'
```

> ⚠️ 脚本开箱即用前请先修改头部默认配置：`CONTAINER_NAME`、`PORT_MAP`、`MODEL_NAME`、`SERVED_NAME`、`GPU_PIN`、`VLLM_DIR`（脚本内有注释指引）。

---

## 4. 参数说明

运行 `./vllm_run -h` 可查看完整帮助。常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--container_name` | `vllm-user` | Docker 容器名 |
| `--image_name` | 本地 tag | vLLM 镜像（qwen4 模型自动切换专用镜像） |
| `--port_map` | `8000:8000` | 端口映射 `宿主机:容器` |
| `--model_name` | 本地 HF 目录 | 模型路径（绝对路径自动只读挂载进容器） |
| `--served_name` / `--display_model_name` | `Qwen-3.8-Flash-Next` | API 对外暴露的模型名 |
| `--gpu_pin` | `4,5,6,7` | 手动指定 GPU（空 = 自动检测空闲卡） |
| `--tp` / `--dp` | `4` / `1` | 张量并行度 / 数据并行度（TP×DP 必须等于 GPU 数） |
| `--gpu_mem_util` | `0.90` | 显存利用率上限 |
| `--max_model_len` | `262144` | 最大上下文（输入+输出 tokens） |
| `--max_num_seqs` | `2` | 最大并发请求数 |
| `--max_batched_tokens` | `4096` | prefill 分块大小 |
| `--eager_mode 1` / `--eager` | 0 | 跳过 CUDA Graph 编译快速启动（1~2 分钟就绪，性能略降） |
| `--kv_cache_memory` | `8400000000` | 钉死 KV 池大小（字节；bf16 下 ≈573K token，盖住 2×256K） |
| `--mamba_ssm_dtype` | `bfloat16` | 线性注意力（GDN）状态精度（省一半状态内存） |
| `--moe_backend` | `auto` | MoE 后端（A40 实测自动选 MARLIN） |
| `--ple_cpu_offload` | `1` | 51B n-gram 表（~48GiB）卸载到主机内存（4C 必开） |
| `--distributed_executor_backend` | `mp` | 分布式执行后端（PLE 卸载 worker 需要 mp） |
| `--timeout_seconds` | `1800` | 等待服务就绪超时（首次 PLE 搬运需 5~15 分钟） |
| `--shm_size` | `64g` | 容器共享内存（多卡 TP 通信） |
| `--api_key` | 空 | API 访问密钥（可选） |

---

## 5. 使用示例

```bash
# 默认部署（4 卡：GPU 4-7、PLE 卸载、TP4、0.90 显存、8000 端口）
./vllm_run

# 显式指定常用项
./vllm_run \
  --model_name /home/user/Project/vllm/models/Qwen3.8-Flash-Next-FP8 \
  --served_name Qwen-3.8-Flash-Next \
  --gpu_pin "4,5,6,7" \
  --port_map 8000:8000 \
  --gpu_mem_util 0.90

# 快速启动（跳过 CUDA Graph 编译，1~2 分钟就绪）
./vllm_run --eager

# 部署其他模型（脚本自动识别类型：dense 模型不加 EP，qwen4 自动加专属参数）
./vllm_run --model_name /path/to/any/model --served_name my-model --gpu_pin "0,1" --tp 2
```

---

## 6. 部署实战总结（4 卡落地配置）

### 6.1 最终生效的 vLLM 参数

```bash
vllm serve /home/user/Project/vllm/models/Qwen3.8-Flash-Next-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen-3.8-Flash-Next \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-expert-parallel \
  --moe-backend auto \
  --mamba-ssm-cache-dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --kv-cache-memory-bytes 8400000000 \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

容器级（docker run）：`-e VLLM_PLE_CPU_OFFLOAD=1`、`-e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800`、
`-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`--cap-add SYS_PTRACE`、
`NCCL_P2P_DISABLE/IB_DISABLE/CUMEM_DISABLE=1`（无 IB 走 TCP）、`--shm-size 64g`、`--ipc=host`、`--restart always`。

### 6.2 参数逐项说明

| 参数 | 值 | 为什么 |
|---|---|---|
| `--tensor-parallel-size 4` | 4 | kv_heads=4 必须整除 TP（只能 1/2/4） |
| `--enable-expert-parallel` | 开 | 512 专家按 EP 切分（128/卡），block-128 FP8 必需 |
| `--moe-backend auto` | auto | A40（sm_86）实测自动选 **MARLIN**；triton 后端不支持 block-FP8 量化组合 |
| `--kv-cache-dtype bfloat16` | bf16 | **QSA 硬性要求**（fp8 主 KV cache 直接 NotImplementedError） |
| `--kv-cache-memory-bytes 8400000000` | 8.4GB | 钉死 KV 池 = bf16 下 **573,741 tokens（2.19x，盖住 2×256K）**；原 9.6GB 每卡仅剩几十 MiB，推理期 OOM（见踩坑 #7） |
| `--mamba-ssm-cache-dtype bfloat16` | bf16 | GDN 状态精度；config 默认 float32，bf16 状态减半 |
| `--distributed-executor-backend mp` | mp | PLE 卸载 worker 通信需要（默认 ray 不支持） |
| `--speculative-config` | mtp×3 | MTP 投机解码（checkpoint 自带 1 层 MTP） |
| `--max-num-seqs 2` | 2 | 低并发速度优先；2×256K 上下文刚好在 KV 池内 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | — | 消除分配器碎片化（崩溃时白丢 291~326MiB） |
| `--cap-add SYS_PTRACE` | — | **expandable_segments 的必需配套**：PLE 卸载跨进程共享 CUDA tensor 走 `pidfd_getfd`，默认无 CAP_SYS_PTRACE 会 EPERM（见踩坑 #8） |

### 6.3 显存预算（每卡，44.42 GiB 可用）

| 项 | 大小 |
|---|---|
| 权重（(172.78−48)/4，PLE 已卸载） | ~31.3 GiB |
| KV cache（8.4GB / 4 卡） | ~2.1 GiB（池总 573K tokens） |
| torch/CUDA/CUDA Graph 等开销 | ~2~4 GiB |
| **每卡总占用** | **~44.3 GiB（实测），余量 ~1.2 GiB** |
| 主机内存（PLE 表 + 缓冲） | ~48~72 GiB |

---

## 7. 实测效果

| 检查项 | 实测结果 |
|---|---|
| API 就绪 | `/v1/models` 200 OK，`"id":"Qwen-3.8-Flash-Next","max_model_len":262144` |
| PLE 卸载 | 日志 `Registrations complete (dp_size=1, tp_size=4, layers=['...ple_embedding'])` + `Busy-loop started`；RAM 增长 ~48~72GiB |
| KV 容量 | `GPU KV cache size: 573,741 tokens, Maximum concurrency for 262,144 tokens per request: 2.19x` |
| 显存 | 4 卡各 **44.3 GB / 46 GB（0.96）**，推理利用率 88~92% |
| 启动时间 | 首次 ~15 分钟（PLE 表搬运 ~9min + 权重 + 编译）；二次启动（页缓存热读）**~6.5 分钟**；`--eager` 3~5 分钟 |
| MoE 后端 | `Using MARLIN Fp8 MoE backend`（A40 自动选择） |
| 冒烟推理 | chat + reasoning 输出正常，`system_fingerprint` 含 `tp4-ep` |
| 稳定性 | 修复 OOM/pidfd 问题后 `RestartCount=0`（详见踩坑 #7/#8） |

---

## 8. 性能与时延

### 8.1 实测（2026-08-30，单流，共享服务器，短文本）

| 模式 | 输入 tokens | 输出 tokens（含 reasoning） | 端到端耗时 | 端到端吞吐 |
|---|---|---|---|---|
| 非流式 #1 | 65 | 74 | 2.20 s | ~34 tok/s |
| 非流式 #2 | 65 | 114 | 1.60 s | ~71 tok/s |
| 非流式 #3 | 65 | 127 | 1.65 s | ~77 tok/s |
| 流式（max_tokens=300） | 66 | 246 | 2.88 s | ~85 tok/s |

> 说明：vLLM 非流式接口为整包缓冲返回，上表为**端到端**（prefill + reasoning + 生成 + 网络）口径；
> 流式接口首字延迟（TTFT）约 1~2 s 量级。实测值受共享服务器、输入/输出长度、prefix cache 命中影响。

### 8.2 预估（调研文档粗估，未正式 benchmark）

| 场景 | 预估 |
|---|---|
| 4C 单流 decode（MTP3，接受率 ~3.5-4 token/step） | **~150-300 tok/s** |
| 4C 64 并发聚合 | ~1.5-2.5K tok/s |
| 8C 单流 / 8C 256 并发聚合 | ~300-700 tok/s / ~5-9K tok/s |
| BF16（不量化） | 约为 8 位方案的一半 |

> 正式基准（`vllm bench serve` 并发 × 输入 × 输出矩阵，MTP 开/关对照）在实验计划中定义（T1-T8），尚未执行；
> 上表为按 6B 激活权重流量（8 位 ≈6.2GB/token）+ 8×A40 合计 ~5.57TB/s 带宽 + EP/MTP 开销的粗估，务必以实测为准。

### 8.3 社区参考数据

- 2×DGX Spark（SGLang + MTP4，NVFP4 版）：~52 tok/s 单流（tonyd2wild fleet-deploy 报告）；
- RTX PRO 6000（单卡 96GB，NVFP4）：74.4 tok/s 单流、483.8 tok/s @32 并发（primitive-ai 实测）。

---

## 9. 踩坑记录（8 条，务必阅读）

1. **参数名错误**：该镜像 vLLM 版本参数为 `--mamba-ssm-cache-dtype`（不是 `--mamba-ssm-dtype`）与 `--kv-cache-memory-bytes`（不是 `--kv-cache-memory`）。写参数前先在镜像里 grep `arg_utils.py`。
2. **`--moe-backend triton` 在 A40 上不可用**：vLLM 对 block-FP8 的 Triton 偏好只在 Hopper（sm_90）生效；A40 必须 `--moe-backend auto`（实测自动落到 MARLIN）。
3. **`--kv-cache-dtype fp8` 被 QSA 硬性拒绝**：`NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache`。必须 bf16，并把 KV 预算翻倍保持容量。
4. **`--no-enable-flashinfer-autotune` 不存在**：该参数组是 `--enable-flashinfer-autotune`（store_true，默认关），直接删掉即可。
5. **`VLLM_PLE_CPU_OFFLOAD=1` 必须配合 `--distributed-executor-backend mp`**：默认 ray executor 不支持 PLE 卸载 worker。
6. **镜像选择是硬性要求**：qwen4_exp 架构必须用 day-0 专用镜像，官方 nightly 会报 `pydantic ValidationError: model type qwen4_exp but Transformers does not recognize this architecture`。
7. **推理期 CUDA OOM 崩溃（显存零余量）**：上线后 2 次崩溃，均发生在 **2 请求并发 prefill** 时 PLE short-conv 需要多分配 66~80 MiB、而每卡只剩 22~70 MiB。根因：权重 + 9.6GB KV + 开销 = 44.05/44.42 GiB，零余量 + 291~326 MiB 碎片化。修复：KV 9.6GB→**8.4GB**（+1.2GiB 余量/卡）+ `expandable_segments`。**经验：压满显存（>0.97）的配置上线前至少要用 2 并发 × 长输入压测一轮；"能启动 + 短冒烟通过" ≠ 配置安全。**
8. **`expandable_segments` 触发 `pidfd_getfd: Operation not permitted`**：expandable_segments 下 PyTorch 改用 CUDA VMM 分配，跨进程共享 CUDA tensor 走 `pidfd_getfd` 系统调用，Docker 默认无 `CAP_SYS_PTRACE` → PLE 注册阶段 EPERM → 启动失败崩溃循环。修复：docker run 加 **`--cap-add SYS_PTRACE`**。

---

## 10. 故障排查

**症状 → 处置速查**

| 症状 | 第一反应 | 再不行 |
|---|---|---|
| 启动 OOM（权重阶段） | 查日志 `Model loading took` 单卡值，确认是否反量化（正常 ≈31GiB/卡；飙到 60GiB+ = 反量化异常） | 停容器查配置 |
| 启动/推理 OOM（KV 阶段） | 按降级链：`--kv_cache_memory 8400000000 → 7200000000` → `--max_num_seqs 2 → 1` → MTP `3 → 2 → 1` → `--gpu_mem_util 0.90 → 0.85` | 记录日志上报 |
| `pidfd_getfd: Operation not permitted` | 确认 docker run 带 `--cap-add SYS_PTRACE` | 备选 `--security-opt seccomp=unconfined`（更宽，不必要） |
| PLE 卸载 worker 超时/挂死 | 确认 `--distributed-executor-backend mp` 在位；`free -h` 盯 48GiB 表落地 | 查 `/dev/shm`（已设 64g） |
| 服务起但推理极慢 | `nvidia-smi dmon` 看 PCIe 是否打满、单卡是否掉速 | 降并发复测 |
| 容器崩溃循环 | `docker logs --tail 60 <容器>` 看根因；修复后 `./vllm_run` 一键重建（脚本自动 stop/rm 旧容器） | — |

**关键日志锚点（排障用）**

```text
# 模型识别成功
[vllm-run] 模型识别：arch=Qwen4ExpForConditionalGeneration type=qwen4_exp experts=512 mtp=yes moe=1 qwen4=1

# MoE 后端选择（A40 关键行）
INFO [fp8.py:411] Using MARLIN Fp8 MoE backend out of potential backends: [...]

# KV cache 容量
INFO [kv_cache_utils.py:2258] GPU KV cache size: 573,741 tokens, Maximum concurrency for 262,144 tokens per request: 2.19x

# PLE 卸载完成
INFO [worker.py:576] Registrations complete (dp_size=1, tp_size=4, layers=['language_model.model.layers.1.ple.ple_embedding'])
INFO [worker.py:590] Busy-loop started.

# 服务就绪
INFO:     Application startup complete.
```

---

## 11. 许可证

**MIT License with Non-Commercial Clause（MIT-NC）**

本项目基于 MIT 开源协议发布，并**附加禁止商业使用条款**：

- ✅ 允许：个人/学术/研究使用、修改、再分发（需保留版权声明与本许可文本）；
- ✅ 允许：注明出处的引用与教学使用；
- ❌ **禁止任何形式的商业用途**（包括但不限于：商业产品/服务集成、商业 API、付费咨询、以盈利为目的的再分发）。

完整许可文本（同仓库根目录 [`LICENSE`](LICENSE) 文件）：

```text
MIT License (with Non-Commercial Clause)

Copyright (c) 2026 Stu-KatoMegumi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Non-Commercial Clause: The Software may not be used for commercial purposes.
Commercial purposes include, but are not limited to: selling the Software or
derivative works, providing paid services or APIs powered by the Software, or
using the Software in a commercial product or service.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> ⚠️ 提示：标准 MIT 协议本身**允许**商用；按作者要求，本仓库附加了 Non-Commercial Clause（即"MIT-NC"），
> 与标准 MIT 不兼容，引用/再分发时请保留本条款。

---

## 12. 免责声明

- 本脚本在 8×A40（无 NVLink）服务器上针对 `Qwen3.8-Flash-Next-FP8` 实测，其他硬件/模型可能需要调整参数；
- vLLM 官方验证矩阵不含 A40（sm_86），本仓库所有 A40 相关结论均为实测/调研所得，不保证其他环境一致；
- 模型权重与镜像版权归其各自作者/组织所有，使用请遵守对应 License（Qwen 为 `qwen-community-1.0`）；
- 性能数字为特定条件下的实测/估算，请以你自己的环境复测为准。

---

*README 生成：2026-08-30 · 关联文档（本地，未随本仓库分发）：部署报告、部署实验计划、8×A40 部署调研*
