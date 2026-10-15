# P2O：环境、机器路径与代码索引

返回[主计划入口](../p2o_agent_implementation_plan.md)。正文保留原章节编号；跨文件的 `§` 引用按主入口的[章节导航](../p2o_agent_implementation_plan.md#章节导航)定位。设计与待做项不代表已实现，当前实现范围见[TODO-00说明](modules/TODO-00.md)。

<a id="section-0"></a>

## 0. 移交背景、范围与路径

<a id="section-0-1"></a>

### 0.1 用户背景、最新要求与执行纪律

用户是 P2O 论文作者之一，已在数学训练和评估上验证方法，现在要求在本仓库为 ALFWorld/WebShop 补充 agentic 实验。Math 源码用于理解算法及恢复逻辑；不能照搬单轮 prompt、reward、优势和任务身份处理。

**当前授权：用户已要求开始实现，并允许使用本机8张GPU调试；每完成一个大TODO或遇到需要讨论的问题必须暂停，等待反馈再继续。**不包含正式长训练授权。最新要求覆盖前版中“直接改原 trainer”“默认精确复现所有随机状态”“固定训练 manifest”“每个原生 epoch 做 GEPA”的建议；这些旧建议不再是当前默认。

已确定的要求：

1. 复制原 `verl/trainer/main_ppo.py` 和 `verl/trainer/ppo/ray_trainer.py` 建立独立P2O入口/trainer；两个消融使用各自的入口与trainer文件（§5.1），不能在同一训练主循环里堆策略分支。新脚本明确调用对应P2O入口。原 GRPO 入口、trainer 和脚本继续原行为。通用算法/工具如 `core_algos.py` 不复制；共享改动必须默认兼容并验证旧路径。
2. 先设计 Env/EnvManager/adapter 的公共接口。主 trainer/GEPA/collector 只调用接口，ALFWorld/WebShop 差异封装在实现内部，不在主循环铺任务分支。
3. TODO 必须按模块严格串行：上一大 TODO 实现、功能/输入输出/数据验证全部通过并汇报后，才考虑下一项。继任者有单机 8 卡 A100，可做 GPU 功能测试和少量受控 update；这不等于授权正式训练。
4. 遇到任何问题或不确定，先阅读论文和 Math 源码定位，**停下来向用户说明并提问，得到答复再继续依赖该问题的工作**。不得自行改变算法/数据语义、降低验收标准或一直干到底。每个大 TODO 完成后停下，汇报成果、关键点、测试和下一步，等待用户继续指示。
5. 每次实现/测试/决策都更新本计划或其链接记录。**整个方案、接口和后续TODO都是动态计划，不是锁死的执行合同。实现时出现任何问题、未预期diff、与源码/论文/本计划不一致，必须立即停下，先把事实、影响、选项写入记录并与用户讨论；用户答复前不得自行修补语义、扩大范围或继续后续TODO。**用户确定后先修改正文和受影响TODO/验收，再继续实现；不能仅在末尾叠加相互冲突的版本。
6. K=8 时以 episode 二元 success 的比例 `≤1/8` 判难；0、10、非法动作惩罚都不直接用于阈值。
7. SimpleMemory 只保留有限历史，模板必须独立持有、每轮重新插入。新入口初始化时自动扩展 prompt 上限：baseline 4096 + 模板预算 2048 = 6144 tokens；原 GRPO 不变。
8. GEPA 距上次完成至少20个原生训练epoch且本收集区间独立难样本≥32时触发；第20个不足则第21、22……逐epoch检查，GEPA完成后重新计20个epoch。在当前收集区间跨epoch积累、全历史另存身份索引；难样本默认保存任务标识与配方/config，不要求保存原训练seed，允许新初始化增加多样性，不要求所有隐藏随机值与原失败实例完全相同。所有候选仍从操作前的初始状态开始。
9. reflection 用训练前固定模型，在 verl/Ray 管理的独立 1 卡组中生成；GEPA 期间 policy 用 N−1 卡，结束恢复 N 卡。不得另起独立 HTTP/vLLM/API 反思服务。
10. GEPA feedback 面向多轮环境，加入轻量规则检测答案泄漏；不用额外 LLM 审查。断点续训必须保存难样本、模板、分配及搜索/资源阶段状态。
11. 样本身份是稳定任务标识（WebShop goal index、ALFWorld gamefile及数据版本命名空间），不含seed、价格等初始化随机值。GEPA只用当前收集区间的难样本；策略1使用全历史曾判难的样本身份，不能TTL淘汰或因后来成功删除。
12. dev按比例取，最低8、最高50，其余全部归train；比例暂建议25%。CRM默认采用B（复用rollout log-prob），生产路径不再额外重算带模板概率。
13. 分别实现历史难样本预先加模板、当前组全失败后重试两种消融；共有环境/GEPA/评分/状态组件复用，控制流程用不同.py。

实际模型位置由用户更新为 `/ceph_home/arknet/hf_models/Qwen/Qwen3-8B`，已重新确认路径存在。本机已核对为8张A100 80GB，预检时每卡占用10 MiB、利用率0%；驱动570.195.03。后续两容器逐卡基础CUDA运算已通过（见下方环境恢复记录）；模型加载仍未验证。

<a id="section-0-2"></a>

### 0.2 工作区与模型路径

本会话工作目录为 `/ceph_home/zhangkaiqi2024/verl-agent`，shell 为 zsh。以下绝对路径是用户本次移交要求保留的机器位置；代码中仍应使用配置及 repo-relative 路径，不把这些机器路径写死在新模块里。

| 用途 | 绝对路径/定位 | 核对状态 |
| --- | --- | --- |
| 目标仓库 `AGENT_ROOT` | `/ceph_home/zhangkaiqi2024/verl-agent` | 存在；实现应在这里做 |
| 参考仓库 `P2O_ROOT` | `/ceph_home/zhangkaiqi2024/P2O` | 存在；当前仅作只读参考 |
| 主计划入口（详细内容按主题拆分） | `/ceph_home/zhangkaiqi2024/verl-agent/docs/p2o_agent_implementation_plan.md` | 原位更新索引；子文档各自维护当前内容 |
| 原 P2O trainer | `/ceph_home/zhangkaiqi2024/P2O/verl/trainer/ppo/ray_trainer_gepa_wo_additional_grpo_in_distribution.py` | 已阅读并与下行比较 |
| 原 Math baseline | `/ceph_home/zhangkaiqi2024/P2O/verl/trainer/ppo/ray_trainer_baseline.py` | 已比较 |
| P2O 启动脚本目录 | `/ceph_home/zhangkaiqi2024/P2O/my_scripts` | 注意脚本名不等于实际 reflection 配置 |
| ALFWorld baseline | `/ceph_home/zhangkaiqi2024/verl-agent/my_scripts/run_alfworld_trajectory_grpo.sh` | source 同目录 `run_alfworld.sh` |
| WebShop baseline | `/ceph_home/zhangkaiqi2024/verl-agent/my_scripts/run_webshop.sh` | 默认 step grouping |
| 后续实现/测试使用的模型 | `/ceph_home/arknet/hf_models/Qwen/Qwen3-8B` | 用户最新指定，路径已确认存在；未加载 |
| 两个原 baseline 脚本内的旧 model_path | `/mnt/shared-storage-user/gpfs1-agent-security/luxinyu1/cot-p2o-space/Qwen3-8B` | 当前不可见；新 P2O 脚本用上一行，不为此改旧脚本 |
| 两脚本当前保存根目录 | `/mnt/shared-storage-user/gpfs1-agent-security/luxinyu1/cot-p2o-space/saved_models/p2o-agent/` | 当前会话不存在；实际运行需检查挂载或配置新的路径 |

`πinit` 的最终路径应取自实际启动训练时解析出的初始模型配置并固定 revision/fingerprint；不得因为上面脚本路径不可用而自动替换成别的模型。用户已经给出上表的实际模型，不再把模型位置作为待提问项；尚未加载模型；训练容器依赖版本和基础GPU运算已核对，模型运行显存留待后续测试。

<a id="section-0-2-1"></a>

### 0.2.1 本机任务容器（用户已确认）

两个容器都在当前这台机器上，分别作为对应任务的运行和调试环境：

| 任务 | 容器名称 | 信息来源与核验状态 |
| --- | --- | --- |
| ALFWorld | `verl_zhangkaiqi2024_alfworld` | 用户指定；运行中，Python 3.12.11，PyTorch 2.8.0+cu128，vLLM 0.11.0，初始化成功，挂载恢复，8卡基础运算通过 |
| WebShop | `verl_zhangkaiqi2024_webshop` | 用户指定；运行中，Python 3.10.12，PyTorch 2.6.0/CUDA 12.4，vLLM 0.8.5.post1，初始化成功，挂载恢复，8卡基础运算通过 |

后续两任务的依赖检查、环境reset/step、模型/GPU功能测试及有界update，应在各自容器中执行，并记录容器内实际Python路径、依赖版本、CUDA/GPU可见性和执行命令。先核对容器状态及仓库、模型、数据、输出目录的挂载映射；上表§0.2的路径为宿主机已知位置，容器内路径以实际挂载为准。代码修改继续保存在目标仓库，确认容器运行的是该工作区的当前代码。

此前宿主机的user-site/PyTorch覆盖诊断仅作历史证据，后续在用户指定容器中执行。用户授权后已删除旧容器，按 `/ceph_home/zhangkaiqi2024/start_docker.sh` 的GPU/网络/shm/挂载参数、各任务指定镜像和原任务容器名称重建；随后分别执行 `my_scripts/init_alfworld.sh` 和 `my_scripts/init_webshop.sh`，最终均exit code=0。WebShop安装时临时使用 `PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL=` 排除镜像附加包源的TLS重试，脚本及版本约束未改。

仓库与指定模型config、任务数据路径已在容器内核对可读；两容器中GPU 0–7逐卡基础CUDA运算、关键依赖/训练入口导入通过。真实环境episode、模型权重加载及vLLM/FSDP功能尚未验证。重建命令、镜像/容器ID、实际依赖版本、初始化与检查日志见本页[环境恢复证据](#recovery-evidence)；旧失败证据保留在预检记录，不再作为当前阻塞。

<a id="section-0-3"></a>

### 0.3 数据和初始化入口

以下路径均相对 `AGENT_ROOT`，移交时用上面的根目录即可还原绝对路径：

| 资源 | 路径与用途 |
| --- | --- |
| ALFWorld parquet | `data/alfworld/text/train.parquet`、`test.parquet`；文件存在，实际行数待训练环境读取 |
| WebShop parquet | `data/webshop/text/train.parquet`、`test.parquet`；文件存在，实际行数待读取 |
| ALFWorld 游戏数据 | `data/alfworld/json_2.1.1/{train,valid_seen,valid_unseen}`；train 路径已确认存在 |
| ALFWorld 逻辑/语言资源 | `data/alfworld/logic/{alfred.pddl,alfred.twl2}`，具体使用配置及 game 文件需随数据版本核验 |
| ALFWorld 环境变量 | `run_alfworld.sh` 设置 `ALFWORLD_DATA=$AGENT_ROOT/data/alfworld` |
| WebShop 当前小商品集 | `data/webshop/items_shuffle_1000.json`、`items_ins_v2_1000.json`；均存在 |
| WebShop 搜索索引 | `data/webshop/search_engine/indexes`；存在。实际索引由 `num_products` 决定，不因 JSON 文件名含 1000 就假定使用 `indexes_1k` |
| WebShop 程序内部数据入口 | `agent_system/environments/env_package/webshop/webshop/data` 和同级 `search_engine/indexes`，当前为存在的符号链接 |
| 环境准备脚本 | `my_scripts/init_alfworld.sh`、`my_scripts/init_webshop.sh`；对应重建容器内已分别执行成功，结果见下方环境恢复记录 |
| 当前公共训练配置 | `verl/trainer/config/ppo_trainer.yaml`，具体配置需再叠加脚本 Hydra override |

当前默认 `env.webshop.use_small=True`、`human_goals=False`、`env.seed=0`。ALFWorld 默认 `eval_in_distribution`，配置文件是 `agent_system/environments/env_package/alfworld/configs/config_tw.yaml`，`domain_randomization=False`、`goal_desc_human_anns_prob=0.0`。这些是阅读时的默认值，不覆盖用户未来显式 override。

<a id="section-0-4"></a>

### 0.4 接手者的代码阅读索引

| 代码（相对 `AGENT_ROOT`） | 优先核对的函数/职责 |
| --- | --- |
| `verl/trainer/main_ppo.py` | `TaskRunner.run`：环境/worker/resource pool/reward manager/collector 装配 |
| `verl/trainer/ppo/ray_trainer.py` | `fit`、`compute_advantage`、`init_workers`、checkpoint、验证与 epoch 结束 |
| `agent_system/multi_turn_rollout/rollout_loop.py` | `preprocess_single_sample`、`vanilla_multi_turn_loop`、`gather_rollout_data`、`multi_turn_loop` |
| `agent_system/multi_turn_rollout/utils.py` | `adjust_batch` 的复制 padding |
| `agent_system/environments/env_manager.py` | 两任务 manager 的 `reset/step/build_text_obs/_process_batch`、`make_envs` |
| `agent_system/environments/env_package/alfworld/envs.py` | `AlfworldWorker`、`AlfworldEnvs`、`compute_reward` |
| `agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py` | 游戏收集、`init_env`、`register_games`、wrapper |
| `agent_system/environments/env_package/alfworld/alfworld/agents/utils/misc.py` | `Demangler`、任务描述随机性 |
| `agent_system/environments/env_package/webshop/envs.py` | worker 构造/`reset(idx)`、group seed、训练/验证 goal 采样 |
| `agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py` | `WebAgentTextEnv`、`SimServer`、`SimBrowser`；真正运行的本地模拟环境 |
| `agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py` | `load_products`、`generate_product_prices`、检索随机分支 |
| `agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py` | 随机价格上限、goal 构造、`get_reward` |
| `agent_system/memory/memory.py` | `SimpleMemory.reset/store/fetch`；必须清空历史后重放 |
| `agent_system/reward_manager/episode.py` | episode reward 到 turn token 的写入 |
| `verl/trainer/ppo/agent_diagnostics.py` | `compute_agent_grpo`，不要用 Math advantage 替换 |
| `verl/workers/actor/dp_actor.py`、`verl/trainer/ppo/core_algos.py` | 训练输入字段选择、PPO loss、CRM 接入 |
| `verl/workers/fsdp_workers.py`、`verl/single_controller/ray/base.py` | 固定 world size、worker group、placement group 生命周期 |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | N 卡分片与 HF 导出、optimizer/scheduler/RNG 恢复 |

`docs/grpo_alfworld_webshop_call_chain.md` 可辅助定位，但其中部分参数和 WebShop seed 描述早于当前源码，本计划已按源码更正，接手者不能直接把旧文档数字当成当前事实。执行前重新检查 `git status` 和相关 `AGENTS.md`；最初只读规划时尚无 P2O agent 模块，当前已有 TODO-00 合同与 Protocol，详见[实现说明](modules/TODO-00.md)。

<a id="recovery-evidence"></a>

## 环境恢复证据

用户授权后已删除两个仅运行sleep的旧容器，复用 `/ceph_home/zhangkaiqi2024/start_docker.sh` 的docker run参数，分别设置任务容器名和镜像：

```bash
docker run -d --runtime=nvidia --gpus all --net=host --shm-size=100g --cap-add=SYS_ADMIN \
  -v /ceph_home/:/ceph_home -v /141nfs/:/141nfs/ \
  --name verl_zhangkaiqi2024_alfworld --entrypoint /bin/bash \
  verlai/verl:vllm011.latest -c 'sleep infinity'
docker run -d --runtime=nvidia --gpus all --net=host --shm-size=100g --cap-add=SYS_ADMIN \
  -v /ceph_home/:/ceph_home -v /141nfs/:/141nfs/ \
  --name verl_zhangkaiqi2024_webshop --entrypoint /bin/bash \
  whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3 -c 'sleep infinity'
```

新容器ID分别为 `378227a4f4fd`、`2c9b47f1ca56`。镜像指纹分别为 `bb140137cf42cdb18990695fed2ec53cd5fdf69dfaad2ce2e85328031a8c8c6f`、`e71ed32d35f6c310db7709e303b5954eb05f5cbdd7cf7073ecf74de1d44322d1`。重建后仓库、指定Qwen3-8B config及数据可读，P2O-ENV-002解除。旧容器失效挂载的底层原因未定，不以重建成功推断原因。

初始化成功命令：

```bash
docker exec -w /ceph_home/zhangkaiqi2024/verl-agent verl_zhangkaiqi2024_alfworld \
  bash -e my_scripts/init_alfworld.sh
docker exec -e PIP_CONFIG_FILE=/dev/null -e PIP_EXTRA_INDEX_URL= \
  -w /ceph_home/zhangkaiqi2024/verl-agent verl_zhangkaiqi2024_webshop \
  bash -e my_scripts/init_webshop.sh
```

两个最终命令exit code均为0。WebShop前两次安装在镜像附加源 `pypi.ngc.nvidia.com` 反复TLS失败；终止了本次任务的对应pip进程后，使用上述临时环境变量忽略pip配置中的附加源重跑。脚本内原版本约束、主镜像源和代理配置保持原样，未改初始化脚本。WebShop安装JDK后javac为11.0.32.1，脚本末尾的spaCy/训练入口/WebAgentTextEnv导入及train=16/test=128检查通过。

| 项目 | ALFWorld容器 | WebShop容器 |
| --- | --- | --- |
| torch / CUDA | 2.8.0+cu128 / 12.8 | 2.6.0 / 12.4 |
| ray / vllm | 2.49.2 / 0.11.0 | 2.46.0 / 0.8.5.post1 |
| transformers | 4.57.1 | 4.51.1 |
| 任务依赖 | gymnasium 0.29.1、stable-baselines3 2.6.0、alfworld 0.4.2、textworld 1.7.0 | gym 0.24.0、Flask 2.1.2、Werkzeug 2.1.0、pyserini 0.17.0、faiss-cpu 1.7.4 |
| 其他核对 | tensordict 0.10.0 | spacy 3.7.5、en-core-web-sm 3.7.1、tensordict 0.8.3、datasets 4.0.0、wandb 0.22.3 |
| 挂载/导入 | 实际仓库路径、模型config、ALF数据及训练入口/AlfworldWorker导入通过 | 实际仓库路径、模型config、WebShop数据/索引目录及训练入口/WebAgentTextEnv/spaCy导入通过 |
| GPU基础运算 | GPU 0–7逐卡 `arange(4).square().sum()==14` 通过 | 同左 |

GPU运算是环境烟雾检查，未加载模型、未做FSDP/vLLM生成或训练update。WS索引路径与相关依赖导入通过不等于真实检索episode通过；真实环境行为仍按TODO-02验收。

原始日志位于本机 `/tmp/p2o-todo00-20261015/`：`init_alfworld.log`、`init_webshop_final.log`、`environment_alfworld.log`、`environment_webshop.log`、`contracts_alfworld.log`、`contracts_webshop.log`。前两次中断日志为 `init_webshop.log`、`init_webshop_retry.log`；临时日志可能被系统清理，本记录保存了命令、版本与关键结果。

TODO-00实现验收时的收尾检查：`git diff --check`通过；所有文档相对链接、代码围栏、已记录源码hash和样例outcome核对通过；计划仅TODO-00勾选，剩余14项未勾选。原入口/trainer、环境/collector/reward/actor/core_algos和两个init脚本的git diff为空。GPU测试进程退出后8卡均恢复占用10 MiB、利用率0%，两个任务容器保持running。
