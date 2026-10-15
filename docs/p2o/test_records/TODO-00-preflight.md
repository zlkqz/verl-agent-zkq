# TODO-00 预检记录（未验收）

日期：2026-10-15，Asia/Harbin，约11:30–11:40。工作目录：`/ceph_home/zhangkaiqi2024/verl-agent`。

本记录只证明路径、版本、设备发现及只读代码核对，不代表合同模块、真实环境、模型生成或GPU计算测试通过。问题P2O-ENV-001见[实施日志](../implementation_log.md)。

## 仓库与文件

运行 `git status --short`、两仓库的 `git rev-parse --short HEAD`，起始结果：

```text
目标仓库 HEAD: a4f930c
参考仓库 HEAD: 92997e2
?? docs/p2o_agent_implementation_plan.md
```

检查目标仓库全部文件及根路径祖先，未找到适用的AGENTS.md。参考P2O仓库只读。原计划归档commit：`0d4999d`。

模型目录 `/ceph_home/arknet/hf_models/Qwen/Qwen3-8B` 存在，包括config、tokenizer及5个safetensors分片；未加载或校验权重内容。使用现有verl解释器和 `pyarrow.parquet.read_metadata` 读取行数：

| 路径 | 行数 |
| --- | ---: |
| data/alfworld/text/train.parquet | 16 |
| data/alfworld/text/test.parquet | 128 |
| data/webshop/text/train.parquet | 16 |
| data/webshop/text/test.parquet | 128 |

已核对ALFWorld游戏数据目录存在；真实环境启动、WebShop索引加载等留给后续TODO验收。

## GPU资源

命令：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

结果：GPU 0–7均为NVIDIA A100 80GB PCIe，81920 MiB，每卡占用10 MiB、利用率0%；驱动均为570.195.03。未启动其他任务或分配显存做模型计算。

## Python与CUDA诊断

当前shell `command -v python` 无结果，`python3` 为 `/usr/bin/python3`（3.10.12）；其metadata中未安装ray、vllm、transformers、omegaconf、pyarrow、gym、gymnasium、tensordict。

对两个已有conda解释器分别运行如下脚本（先不加环境变量，再加 `PYTHONNOUSERSITE=1` 对照；CUDA探测在默认verl导入和两套禁用user-site的命令中执行）：

```python
import sys, site
from importlib import metadata
print(sys.executable, site.ENABLE_USER_SITE)
for name in ('torch', 'ray', 'vllm', 'transformers', 'pytest', 'tensordict', 'pyarrow'):
    try:
        dist = metadata.distribution(name)
        print(name, dist.version, str(dist.locate_file('')))
    except metadata.PackageNotFoundError:
        print(name, 'not installed')
import torch
print(torch.__file__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
```

| 项目 | 默认verl环境 | 禁用user-site的verl | 禁用user-site的verl_newer_vllm |
| --- | --- | --- | --- |
| Python路径 | miniconda3/envs/verl/bin/python | 同左 | miniconda3/envs/verl_newer_vllm/bin/python |
| torch | 2.14.0，用户目录覆盖 | 2.6.0 | 2.8.0 |
| CUDA runtime | 默认失败，未单独读取runtime版本 | 12.4 | 12.8 |
| CUDA可用/设备数 | 导入verl时CUDA初始化失败 | True / 8 | True / 8 |
| ray | 2.53.0 | 2.53.0 | 2.53.0 |
| vllm | 0.8.5.post1 | 0.8.5.post1 | 0.11.0 |
| transformers | 4.57.1 | 4.57.1 | 4.57.1 |
| pytest | 9.1.1 | 9.0.2 | 9.0.2 |
| tensordict | 0.6.2 | 0.6.2 | 0.6.2 |
| pyarrow | 22.0.0 | 22.0.0 | 22.0.0 |

路径均相对 `/ceph_home/zhangkaiqi2024/`。默认torch实际来源：`/ceph_home/zhangkaiqi2024/.local/lib/python3.10/site-packages/torch/`。禁用user-site后torch来自对应conda的 `lib/python3.10/site-packages/torch/`。

默认导入verl时的关键输出：

```text
CUDA initialization: The NVIDIA driver on your system is too old (found version 12080).
```

诊断仅临时设置环境变量，不修改包、shell配置或原baseline脚本。推荐候选执行命令（待用户确认）：

```bash
PYTHONNOUSERSITE=1 /ceph_home/zhangkaiqi2024/miniconda3/envs/verl/bin/python ...
```

完整后端及环境依赖兼容性仍未验证；CUDA设备发现不等于计算、模型加载或8卡训练通过。

## 接口源码核对

- ALFWorld worker的reset返回obs/infos，step返回obs/scores/dones/infos；向量wrapper将info解包，纯文本奖励为 `10 * won`，含 `extra.gamefile`、`admissible_commands`、`observation_text` 等任务/观察字段。manager当前reset不透传kwargs。
- WebShop worker的reset接收goal index，返回obs/info；step保存原始 `task_score`，按 `done and task_score == 1.0` 生成won及10/0奖励。底层terminal step会自动reset，终止身份/动作快照必须由P2O后续adapter处理。
- 两manager当前没有P2O subset reset/step合同；这是计划已知边界，TODO-00定义能力、TODO-02实现，不在本轮修改原环境代码。
- Math原判难按非零reward及prompt signature，不直接迁移；已定位/复核其判难、GEPA终结及checkpoint恢复入口，合同继续按稳定任务身份及episode success定义。

未执行pytest、真实环境reset/step、模型加载、GPU计算或训练；TODO-00的所有实现与验收工作仍待继续。

记录收尾检查通过：`git diff --check`；三份Markdown的相对文件链接、代码围栏配对及行尾空白检查；15个TODO保持未勾选且顺序完整。这些文档检查不计入TODO-00合同验收。
