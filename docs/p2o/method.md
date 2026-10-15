# P2O：方法语义、baseline与两种策略

返回[主计划入口](../p2o_agent_implementation_plan.md)。正文保留原章节编号；跨文件的 `§` 引用按主入口的[章节导航](../p2o_agent_implementation_plan.md#章节导航)定位。设计与待做项不代表已实现，当前实现范围见[TODO-00说明](modules/TODO-00.md)。

<a id="section-1"></a>

## 1. 建议与可行性结论

P2O 可以迁移到当前 agent 训练框架。实现必须同时包含：可重复访问的困难任务、GEPA 模板搜索与分配、逐轮上下文蒸馏、CRM，以及训练/搜索阶段的状态管理。只把 GEPA 找到的提示插入 rollout，不是完整的 P2O。

**建议采用阶段切换实现你的 GPU 安排：正常训练用 N 张卡；达到 GEPA 周期边界后冻结当前 policy，切换为 N−1 卡 policy 推理 WorkerGroup 和 1 卡初始模型反思 WorkerGroup；GEPA 结束后销毁两个临时组，恢复 N 卡训练 WorkerGroup。**

这要求实现 worker 生命周期管理，但不要求 policy 在 GEPA 期间更新参数。保存并恢复 N 卡原始训练状态，临时 policy 组只加载当前权重做推理，可以避免 optimizer 的 N→N−1→N 重分片。以脚本中的 8 卡为例，就是 `8 卡训练 → 7 卡当前 policy 推理 + 1 卡初始模型反思 → 8 卡训练`。

当前仓库没有支持上述切换的完整接口，也不支持直接把一个活着的 FSDP 组减少一个 rank。以下可行性判断来自代码结构，尚未经过 GPU 实机验证；关键工程验收之一是完整完成一次资源往返切换且训练状态连续；按TODO顺序在前置模块完成后的TODO-10进行，不据此跳过TODO-00。

<a id="section-2"></a>

## 2. 方法理解及迁移后保持的语义

<a id="section-2-1"></a>

### 2.1 三种模型角色必须区分

| 角色 | 权重 | 用途 |
| --- | --- | --- |
| 当前 policy `πθ` | 持续训练；GEPA 时冻结为 本次 GEPA 周期边界的快照 | 训练轨迹生成、候选模板的环境评估、梯度更新 |
| 旧 policy `πold` | 当前训练 batch 更新前的 policy | 两种上下文的 log-prob，以及 PPO ratio 的分母；可在 update 前完成打分，不必常驻第三份模型 |
| 反思模型 `πinit` | 开始训练前的模型，整个实验固定 | 根据失败反馈生成新的模板；单独的 1 卡 WorkerGroup |

CRM 的分子、分母都来自 `πold`，**不能用 `πinit` 计算 CRM**。现有 `Role.RefPolicy` 面向 KL 的 log-prob 计算，而且两个指定 baseline 默认不启用 KL；反思组应有独立角色与生成接口，不能假定已有 KL ref 组就是反思组。

<a id="section-2-2"></a>

### 2.2 外循环与两种消融

1. 正常从训练集采样任务标识，结合seed等环境随机性生成实例；记录稳定sample_key和本次instance_id。
2. 两种策略分别控制是否带模板及何时重试，见§4.4。每个有效K组按success≤1/8登记难样本；策略2触发重试的条件更严格，为首次无模板组0/8。
3. 自上次GEPA完成后至少20个epoch且本区间去重难样本≥32，冻结当前policy，使用本区间难样本做GEPA。未足32则继续延长本区间并逐epoch检查。
4. 提交本轮模板、dev分数和样本求解证据，更新跨周期索引，清空本区间收集器并重新计20个epoch；历史曾难身份与旧求解模板不清空。
5. 带模板生成的轨迹使用去模板上下文训练及CRM。策略1在生成前查询历史，策略2先无模板生成、全失败后从初始环境重试。
6. 最终验证使用原始agent prompt，不加模板、不启用失败救援重试。GEPA评价轨迹只用于搜索，不直接作为额外GRPO数据。

“原始prompt”包括baseline任务、环境观察、动作格式、可用动作和有限历史，只移除额外模板。依据：[论文§3及附录](https://arxiv.org/html/2603.21877v4)；历史跨周期保留和失败重试是本次用户指定的agent消融，不能声称Math已有完全相同实现。

<a id="section-2-3"></a>

### 2.3 CRM 先说明操作：同一份回答，两种输入打分

前版把“现有评分调用如何复用”和“相对哪一种训练算额外成本”混在一起，容易让读者误以为方法不同。本节统一以**带模板 rollout，并按带模板输入正常做 GRPO**作为成本参照。用户的理解正确：该流程本来就算带模板 old log-prob；P2O 为去模板学习，需要新增无模板 old log-prob。这里是**去掉模板后评分**，不是给无模板评分再引入模板。

以一条有10步的轨迹为例，每一步都有自己的输入与回答：

```text
第 t 步：
原输入 h[t] = 任务说明 + 当前观察/可用动作 + 最近两步历史
增强输入 hz[t] = h[t] + 独立插入的模板
生成一次回答 y[t]，执行动作，进入下一步。

之后固定这份 y[t]，分别问旧 policy：
AUG[t]  = 在 hz[t] 下，这些回答 token 的 log-prob 是多少？
BASE[t] = 在 h[t]  下，同一批回答 token 的 log-prob 是多少？
```

所有10步都要处理；不是只去掉第0步的模板。只去模板，不改观察、历史、回答或已发生的动作，不重新执行环境。评分是已知完整回答的并行前向计算（teacher forcing），不是再自回归生成回答。多个step可以合批/microbatch评分，不需要10个独立RPC。

<a id="section-2-4"></a>

### 2.4 三份概率各自干什么，以及成本归属

| 数值 | 用什么模型、什么输入 | 用途 |
| --- | --- | --- |
| AUG | 更新前旧policy，带模板输入，同一回答 | 与BASE比较，构造CRM |
| BASE | 更新前旧policy，无模板输入，同一回答 | 与AUG构造CRM，同时作PPO ratio分母；算一次后固定 |
| NEW | 正在更新的policy，无模板输入，同一回答 | 作PPO ratio分子，有梯度；这是训练本来要做的forward/backward |

逐token计算：

```text
CRM = 1{ log(0.01) <= BASE - AUG <= log(10) }
PPO ratio = exp(NEW - BASE)
训练的PG损失 = 原PPO逐token PG损失 × CRM
```

AUG与BASE用同一更新前policy，不能用初始反思模型；两者detach。NEW随优化更新而变，不能把BASE当成NEW以省掉训练forward。CRM仍按原response mask/reduction聚合，不用保留token数重新归一化，不改reward/advantage；无模板行CRM=1。

**按用户采用的成本参照：**带模板普通GRPO本来已有AUG评分；严格P2O多一次BASE评分。这次BASE同时服务上下文蒸馏的PPO分母和CRM，不能全算作CRM独占成本。若参照“已经做去模板蒸馏、但不做CRM”的训练，已有BASE，此时新增的是AUG。这是同一份计算账的不同参照，前版没有讲清；之后不再交替使用这两个参照。

如果把当前仓库真正的原版（没有模板）GRPO拿来比较，还存在带模板生成的额外prefill/token成本，不能只用评分次数代表整个训练成本。

<a id="section-2-5"></a>

### 2.5 A/B 两版只改变 AUG 从哪里来

**A版：仅作测试中的数值参照，不作为生产必经阶段。**

1. 带模板生成回答，保留每步两种输入和同一份回答。
2. actor按带模板输入重算AUG——相当于带模板普通GRPO原有的old-log-prob评分。
3. actor按无模板输入计算BASE——相对上述参照新增的评分。
4. 用AUG、BASE形成CRM；训练只输入无模板context，算NEW，以BASE作分母更新。

**B版：用户已选定的默认实现，生成时的分数直接作为 AUG。**

1. 同样带模板生成回答，同时保留生成引擎返回的逐token log-prob，作为AUG候选值。
2. **省掉A版第2步：不再让actor对带模板输入重算AUG。**
3. 无模板BASE照算，CRM照算，无模板NEW的forward/backward照做。

B省的是“生成完后再对增强输入做一次评分”，**没有省掉无模板评分，也没有跳过任何轮的CRM**。生成时模型已计算预测分布，返回被采token的log-prob有提取/存储/传输开销，但无需再完整前向一遍增强输入。

假设batch展平后有100个step，其中40个带模板、60个无模板：

| 流程 | rollout后actor评分工作（不含生成/训练） |
| --- | --- |
| 带模板普通GRPO（成本参照） | 40个增强输入 + 60个普通输入 |
| A版P2O | 40个增强输入 + 100个无模板输入 |
| B版P2O | 100个无模板输入；40个AUG直接取生成返回值 |

所以A比参照新增40个无模板评分；B比A省掉40个增强评分。每个step长度不同，增强输入还更长，不能把140→100行直接称为总训练加速28.6%。两版都只有一次BASE评分，不得算完CRM后再整批重复计算BASE。

当前仓库 `fit` 本来有整批 `compute_log_prob`，实现时可让它一次生成全批BASE，AUG则直接取模板行的rollout返回值（测试参照A才重算）；先后顺序不改变上述账目。Math代码约3594–3742行在CRM里先算base、后面公共block又重算全批base，这个重复可以消除。

**两版优缺点与使用门槛：**

|  | A版 | B版 |
| --- | --- | --- |
| AUG来源 | actor重算 | rollout引擎返回 |
| 优点 | AUG/BASE同一评分实现，易对齐与调试 | 少一次模板step的增强评分，长模板时可能更值得 |
| 代价/风险 | 多一次评分，增加时间及临时内存 | 两引擎概率口径/数值可能不同，影响CRM边界；多保存/传输逐token数据 |
| 是否改变目标公式 | 否 | 意图不变，但只有口径对齐并通过校验才能作为等价优化使用 |

当前vLLM SPMD路径已有 `rollout_log_probs`；`logprobs=0` 在该路径仍提取被采token分数。但必须核对实际安装版本中temperature/top-p/top-k等处理前后的概率口径。actor使用temperature；训练默认0.8/0.95，不能假设两者天然一致，也不能凭一个token的标量随意转换分布。

实现直接做B；验收在独立小batch测试中用actor增强评分作为A参照，比较同一冻结权重/context/response下返回的AUG：覆盖两任务、长短turn、EOS/截断、padding/reorder；测试真实0.8/0.95，1/1仅作定位。报告log-prob误差分位数/最大值、CRM mask不一致率与边界token、loss/梯度差异，以及时间/显存。容差先测重复评分噪声并与用户讨论；校验失败或版本/context/token对不上时停止并向用户报告，修复口径/数据链路；不能在生产路径偷偷回退A而重新引入用户明确要省掉的评分。

A/B都不减少多轮任务的无模板BASE评分。进一步优化先做分桶/microbatch和避免冗余entropy，而不是只给第一轮做CRM或改成轨迹平均比率。跨不同输入不能直接复用KV；前缀和位置变了。功能等价测试不能证明最终成功率不下降，正式效果需要训练对照。

<a id="section-3"></a>

## 3. 原 P2O 代码相对 baseline 增加了什么

已对比相邻 P2O 仓库的 `verl/trainer/ppo/ray_trainer_baseline.py` 与 `ray_trainer_gepa_wo_additional_grpo_in_distribution.py`。AST 比较显示 trainer 新增 39 个方法，修改 `__init__`、`_create_dataloader`、`_save_checkpoint`、`_load_checkpoint`、`fit`；`init_workers` 没有改变。

| 增量 | 原实现位置/行为 | 本仓库处理 |
| --- | --- | --- |
| 配置、状态和截断 | `_init_gepa_cfg`、模板/困难集缓存、反思示例裁剪 | 独立 P2O 配置和有类型的状态对象 |
| 困难样本挖掘 | `_collect_very_hard_samples`：按 uid 统计全零或仅一个非零 reward；raw chat signature 跨 epoch 匹配 | 用环境任务身份与 episode 成功率，不能用原始占位 prompt 或惩罚后的非零 reward |
| 模板应用 | `_apply_prompt_template_to_sample`；按样本、rollout 分配模板 | 在 collector 每一轮构造输入时应用，保持逐轨迹绑定 |
| 搜索评估 | `_evaluate_prompt_on_samples`、批量多模板评估 | 改为完整环境 episode；返回可重放任务信息和结构化 trace |
| GEPA 搜索 | 空模板、候选选择、反思、minibatch 严格改进接受、dev 评分、覆盖集合、模板分配 | 拆成独立搜索器，保留搜索过程，避免 trainer 内堆叠逻辑 |
| 反思生成 | API / 当前模型 / 绕开 Ray 创建本地 vLLM 三条分支 | 本次固定为 Ray 管理的初始模型生成组 |
| 上下文蒸馏 | `_rebuild_batch_input` 替换原 prompt 并重建 position ids | 每一条 turn 保存双上下文，重构所有相关张量并保持 response 不变 |
| CRM | 更新前对双上下文打分、构造 `template_ratio_mask` | 在 actor 和 loss 层明确传递 `context_ratio_mask` |
| checkpoint | `gepa_state.pt`、GEPA 前后恢复标记 | 扩展为可恢复的训练/资源/搜索状态机 |
| 计数 | GEPA 前后调整 global step，dataloader 总步数增加 epoch 数−1 | 分离训练步和搜索计数，GEPA 不推进训练 scheduler |

CRM 并非只改 trainer 就生效。P2O 仓库的 `verl/workers/actor/dp_actor.py` 还负责选择并传递 `template_ratio_mask`，`verl/trainer/ppo/core_algos.py` 在聚合前将它乘到 `pg_losses` 上。本仓库对应两层都需要适配。

有三处应明确选择新实现的规范：

- **选择器差异：**论文附录 Algorithm 4 是按 dev 分数向量进行非支配筛选；原代码主要按“在哪些 dev 样本上达到当前最优”建立候选并加权抽样，两者不完全等价。当前提案按v4非支配定义实现；这是尚需在TODO-08前和用户确认的论文/Math差异，不额外实现未要求的legacy消融，也不能把两个选择器混称等价。
- **预算差异：**原代码前两轮未按后续循环扣费，before/after minibatch、初始 dev 和 reflection 的计费也没有形成统一硬上限。新实现统一计费与调用前预留额度，预算不足时停止，不沿用“再做最后一轮”的超支逻辑。
- **配置含义：**旧 `template_ratio_cliprange_low=0.99`、`high=9` 实际给出 `[0.01,10]`；新配置直接写 lower/upper，减少误读。`replace_prompts_when_log_prob=false` 也不代表不开蒸馏：CRM 分支内部已调用 `_rebuild_batch_input`。

检查的 `my_scripts/deepmath_qwen3_4B_in_distribution_ref_model_reference_replace_hard_clip_0.01-10.sh` 实际启用了 API reflection；文件名中的 reference 不能作为运行身份依据。迁移时应记录解析后的配置与模型路径。

<a id="section-4"></a>

## 4. 当前 agent 仓库的适配边界

<a id="section-4-1"></a>

### 4.1 保留用户指定的两个 baseline

| 项目 | ALFWorld | WebShop |
| --- | --- | --- |
| 脚本 | `my_scripts/run_alfworld_trajectory_grpo.sh`，source `run_alfworld.sh` | `my_scripts/run_webshop.sh` |
| GRPO grouping | trajectory | step |
| penalty mode | joint | joint |
| actor minibatch | 64 | 256 |
| 每批任务数 / 每任务轨迹数 | 16 / 8 | 16 / 8 |
| 最大环境步数 | 50 | 15 |
| GPU / rollout TP | 8 / 1 | 8 / 1 |
| 训练温度 / top-p | 0.8 / 0.95 | 0.8 / 0.95 |
| 非法动作惩罚 | 0.1 | 0.1 |
| 默认 KL / entropy loss | 关闭 / 0 | 关闭 / 0 |

`actor_rollout_ref.rollout.n` 保持 1；任务轨迹数由 `env.rollout.n=8` 控制。主对照逐任务沿用原有 grouping；如果增加 WebShop trajectory-GRPO 实验，需要对应的 baseline，不能把 grouping 变化算作 P2O 收益。

当前纯文本 ALFWorld 的奖励为 `10 × won`；WebShop 仅在终止且原始分数为 1 时给 10，否则 0。WebShop 的连续 `task_score` 保留为评估/反馈信息，主方案不改变训练 reward。`EpisodeRewardManager` 将整条 episode reward 写到该轨迹每轮 response 的最后一个有效 token；它不是每轮新增的成功奖励。

`agent_diagnostics.compute_agent_grpo` 的 trajectory 分组每条轨迹统计一次 task return，并在 joint 模式累计该轨迹非法动作惩罚；step 分组则按有效 turn 行统计，因此受轨迹长短影响。P2O 调用现有实现，不另写一个 Math 版本 advantage。

<a id="section-4-2"></a>

### 4.2 输入、身份和异步

- `TrajectoryCollector.preprocess_single_sample` 使用当轮 `obs['text']` 新建 user message，原 parquet 的 `raw_prompt` 不会原样成为每轮输入。在 `fit` 入口改 prompt 会被覆盖。
- `gather_rollout_data` 将完整轨迹展平为 turn 行；`uid` 是当前采样组，`traj_uid` 是本次轨迹，二者均为运行时 UUID，不能承担跨 epoch 任务身份。
- ALFWorld、WebShop 的 manager 虽接收 `reset(kwargs)`，目前都没有将 kwargs 转交底层 reset。WebShop worker 支持 `reset(idx)`，上层仍随机抽取；ALFWorld worker 目前只有无参数 reset。必须打通显式任务 reset。
- `gather_rollout_data` 把 success 数组聚合成均值后写入每行。困难挖掘需要在此之前保存每条 episode 的原始 success，不能从均值还原。
- 默认 `rollout.mode=sync`，collector 同步等待 `generate_sequences`；环境内部用 Ray remote + `ray.get`，ALFWorld 底层还使用异步 TextWorld。当前入口有 async worker，但 `fit` 中 async manager 的直接调用被注释，不能宣称已具备完整异步 agent rollout 支持。

首版以用户脚本实际使用的 sync 生成链路接入。所有记录仍携带显式 request/task/trajectory/turn/policy-version 标识，所有阶段切换等待请求完成；真正 async backend 的端到端接入作为单独验收项，不把“多轮环境并行”当作“异步生成已验证”。

<a id="section-4-3"></a>

### 4.3 收集区间与dev比例

确定配置：至少20个完整epoch，**本收集区间**至少32个去重难样本；达到32不是立即触发的充分条件。当前epoch约一个batch，继任者核对parquet后不得暗改单位。

```yaml
gepa:
  interval_epochs: 20
  min_unique_hard_samples: 32
  dev_fraction: 0.25          # 建议起步比例
  dev_min_samples: 8
  dev_max_samples: 50
  reflection_minibatch_samples: 2
  beam_width: 2
  train_variants_per_sample: 2
  dev_instances_per_sample: 1
crm:
  augmented_score_source: rollout   # B，生产不重算增强侧
```

设上次GEPA提交位置为a，已完成训练epoch为e；当前区间为(a,e]。每个epoch结束，先收集本epoch判难记录，再检查 `e-a>=20 and len(interval_hard)>=32`。本区间内曾满足判难条件的sample_key加入集合，同key重复只算一次，后面成功不抹掉该区间曾判难事实；保留判难证据与最新观测分别记录。正常完成GEPA并原子提交后a=e，本区间归档，新建空interval_hard。正常无改进/预算耗尽仍可完成提交；崩溃或未开始就预算不足不重置锚点。

例：epoch12已有32个，也等到20；epoch20已收集80个，则使用全部80个，而非截取32个。epoch20只有30、21只有31、22达到32，则22做GEPA，下轮最早42；这次区间就是22个epoch，不能只取最后20个而丢掉前2个。后续区间是否足32只看新interval_hard，历史索引再大也不充数。曾在旧区间出现的样本若本区间重新被判难，可以再次进入。

设本轮区间总数H：

```text
D = min(50, max(8, floor(H * dev_fraction)))
search-dev = D个不同sample_key
search-train = 剩余全部H-D个sample_key
```

配置校验必须保证 `0<dev_fraction<1`、`1<=D<H`；用户以后覆盖阈值/比例造成空train时明确报错并讨论，不静默改变比例。

25%示例：H=32→24/8；H=80→60/20；H=128→96/32；H=200→150/50；H=240→190/50（train/dev）。不丢掉超过32或超过dev上限的样本；删除旧提案的hard bank容量128/TTL50限制，不能让其截断本区间或遗忘历史。大trace可落盘、分页/惰性加载，精简索引必须完整保存。

按sample_key固定排序/分层划分并保存；同sample_key的所有实例必须在同侧。train可产生合法新初始化，dev固定实例和greedy解码，见§6.2。dev只用于选择，不把其轨迹发给reflection。8只是下限，样本少时分数粒度粗，不通过复制实例虚增独立样本数。训练/最终验证协议保持原样。

功能试运行预算暂用256个policy-eval episodes、16个reflection calls，正式预算待实测讨论。H变大不意味着每次反思全跑train：按minibatch使用完整train池；dev上限50约束每个候选评估成本。预算需涵盖空模板dev、旧模板重评及至少一次有效比较；不足时不能偷偷删样本，须记录并讨论预算。GEPA不会增加训练global_step，最后无后续update时默认不做无用搜索。

<a id="section-4-4"></a>

### 4.4 两种模板使用策略：不同.py实现

**共享状态必须拆开：**

- `interval_hard`：仅本区间曾判难sample_key与证据，供下一轮GEPA；提交后归档/清空。
- `ever_hard`：全训练历史曾判难sample_key集合，策略1命中依据；永久保留，不因成功/TTL/进入新周期删除。
- `cycle_template_registry`：各GEPA周期选中模板、dev分数、policy/panel身份及求解证据，immutable提交；有历史引用的模板不可随checkpoint清理删掉。
- `latest_solved_cycle[sample_key]`：这个样本最后一次有实际求解证据的GEPA周期。不是“最后参与搜索的周期”，也不是“最近一次全局GEPA”。

**策略1：历史曾难样本在rollout前加模板（historical）。**训练采到sample_key时查ever_hard，未命中则原无模板rollout；命中则选择模板。只被一次GEPA解出，使用那次的解题模板；被多次解出，使用最新一次**解出它**的周期中有求解证据的模板。后来GEPA没解出或没见到它，不覆盖先前求解记录。历史曾难但从未被解出/从未参加GEPA，则从最新已提交GEPA的dev评价模板集合采样。组内可采不同模板，episode内固定。

**用户已确认dev回退按覆盖数加权。**最新周期selected模板中令c_j为模板j在本轮固定dev上成功的不同sample_key数，`p_j=c_j/sum(c)`；重叠覆盖分别计入各模板，不用贪婪选择时的“新增覆盖数”。例如覆盖数6/3/1→概率0.6/0.3/0.1；有正权重时0覆盖模板概率为0，不私加epsilon。所有候选使用同一dev时，按dev均分与按覆盖数归一化等价。这指采模板，不是从dev抽训练任务。sum(c)=0、集合为空或首轮GEPA前没有模板时，建议显式回退无模板并记录，不能除零或宣称有覆盖；这是边界处理建议，不是用户已经指定的均匀采样规则。已求解样本的成功模板子集内，建议同样按其所属周期dev覆盖数加权，若全为0需单独记录并讨论（尤其train成功但dev零分的情况）。

Math源码约2694–2772行：dev样本先找selected集合里覆盖它的候选，无则用全selected；按dev aggregate score加权，train样本直接按全selected加权，其solved_status=False。每轮重建映射，没有实现本次要求的全历史latest_solved索引。新实现保留上述采样思想，增加历史层，不能原样复制覆盖旧记录。

用于历史专属模板索引的“解出”必须同时满足：模板在该周期**最终selected覆盖集合**中，并有 `sample_key/template/cycle/instance/policy/outcome` 成功证据。用户已确认至少一个已评变体成功即可，不要求全部成功。未评价、invalid或全失败不能算解出；成功但落选的候选仅留调试日志，不进入可用的样本专属模板档案。最终集合没有解出该样本的模板，就按dev分数回退采样（历史已保留的成功周期仍按既定最新成功规则选择，见§13.2），不能拿落选模板填补。不要从dev总体均分推断某个未测样本成功。

**策略2：无模板全失败后再带模板重试（failure_retry）。**所有当前样本先做无模板K=8 rollout，不先查历史加模板。完整有效组success_count=0才触发，不要求ever_hard命中；1/8属于判难但不触发救援；invalid/timeout不是0/8。触发后沿用共享模板选择器（有历史求解证据则取最近成功周期，否则最新dev采样），回到**第一次rollout的同一初始实例**再做K条增强rollout，逐轮模板保留，B版CRM与去模板训练。不能从失败的终止状态继续，也不能换价格/goal后称为同一样本救援。适配器需在第一次action前保留可重建初始化材料；普通hard recipe仍轻量，必要的世界快照按共享world暂存。

默认建议每组最多一次救援；救援仍0/8不递归循环；首轮GEPA前无可用模板时保留原无模板组。重试前后policy版本不变，不穿插optimizer update。记录 `attempt_phase=base/rescue`、新traj_uid与共同parent_group/sample_key/instance_id，重复请求不能触发第二次救援。

**用户已确认的数据选择：救援组有效完成后，仅重试K=8条参与更新。**首次失败组只用于触发、判难与feedback，首次及救援的环境/生成成本全部计量；不拼成2K组、不把首次失败组混进本次loss。即使救援仍0/8，也用该有效重试组更新（其中非法动作惩罚可能产生非零advantage）。未触发救援或根本无模板可用时，原K组正常更新；救援发生基础设施错误则按invalid组规则处理，不静默当作任务失败或随意换回首次组。首次task reward全0不代表惩罚advantage为0，舍弃首次组是本消融明确的数据选择。

策略2建议interval_hard/ever_hard以首次无模板组判难记录为准，救援结果独立记录，不覆盖首次失败事实；策略1使用其实际rollout（可能带模板）结果判难。两策略产生的GEPA池可能不同，这是消融本身的影响，要记录裸/增强来源。策略2不要求历史命中，只共享历史作为模板选择信息。
