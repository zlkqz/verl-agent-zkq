# TODO-00 预检记录（未验收）

日期：2026-10-15，Asia/Harbin，约11:30–11:40。工作目录：`/ceph_home/zhangkaiqi2024/verl-agent`。

本记录保留初始预检历史；当前环境恢复及TODO-00完成结果见[TODO-00.md](TODO-00.md)。以下旧失败不代表当前环境状态。问题P2O-ENV-001/002见[实施日志](../implementation_log.md)。

## 用户补充的实际运行环境

用户于2026-10-15明确：ALFWorld使用本机容器 `verl_zhangkaiqi2024_alfworld`，WebShop使用本机容器 `verl_zhangkaiqi2024_webshop`。后续各任务在对应容器内预检和调试；本记录“Python与CUDA诊断”一节来自宿主机，不代表容器状态。容器实测及后续重建结果分别见本记录末节和[当前验收记录](TODO-00.md)。

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

诊断仅临时设置环境变量，不修改包、shell配置或原baseline脚本。当时的宿主机候选执行命令如下，现已由用户指定的两任务容器替代，不作为后续默认执行命令：

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

## 2026-10-15 12:50起：指定容器实测

Docker daemon与宿主机均为 `CIP-GPUSERVER-144`；两个容器均为running，默认用户root，bind mount配置为 `/ceph_home:/ceph_home`、`/141nfs:/141nfs`，均RW=true。

| 项目 | ALFWorld容器 | WebShop容器 |
| --- | --- | --- |
| 容器 | verl_zhangkaiqi2024_alfworld | verl_zhangkaiqi2024_webshop |
| 镜像 | verlai/verl:vllm011.latest | whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3 |
| Python | /usr/bin/python3，3.12.11 | /usr/bin/python3，3.10.12 |
| torch | 2.8.0+cu128 | 2.6.0 |
| CUDA runtime / available / count | 12.8 / True / 8 | 12.4 / True / 8 |
| ray | 2.49.2 | 2.46.0 |
| vllm | 0.11.0 | 0.8.5.post1 |
| transformers | 4.57.1 | 4.51.1 |
| tensordict | 0.10.0 | 0.8.3 |
| pytest | 未安装 | 8.1.1 |
| gym / gymnasium | 未安装 / 0.29.1 | 0.24.0 / 未安装 |
| textworld / alfworld | 1.7.0 / 0.4.2 | 未安装 / 未安装 |
| pyarrow | 22.0.0 | 20.0.0 |

metadata版本读取不代表以上各包均已成功import；只import了torch做CUDA设备发现，未执行GPU计算或加载模型。

访问失败复现（将容器名分别取上表两值）：

```bash
docker inspect --format '{{json .Mounts}}' CONTAINER
docker exec -w /tmp CONTAINER ls -ld /ceph_home
docker exec --user 1083:1083 -w /tmp CONTAINER sh -c 'cd /ceph_home/zhangkaiqi2024/verl-agent && pwd'
```

两容器root的 `ls` 均返回 `Permission denied`，UID 1083的 `cd` 也被拒绝。直接 `docker exec -w /ceph_home/zhangkaiqi2024/verl-agent` 的OCI chdir同样失败。宿主机 `ls -ld` 显示 `/ceph_home` 为777、用户目录与仓库为775，UID/GID为1083；仅这些信息不足以确认底层拒绝原因。

依赖/CUDA检查使用 `docker exec -i -w /tmp CONTAINER python3 -u -` 输入本记录中的metadata/torch诊断脚本，不访问不可读挂载。CUDA检查两命令exit code均为0。未改权限、重建容器或实施 `/tmp` 源码暂存提案；问题P2O-ENV-002待用户决定。

后续用户已授权删除旧容器并按指定镜像/启动参数重建、分别执行init脚本，已完成且挂载恢复；本节的旧容器已删除。当前状态和完整重建/初始化/测试证据见[TODO-00验收记录](TODO-00.md)。
