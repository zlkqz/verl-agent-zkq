# TODO-00 实现说明：现在有什么，如何工作

更新：2026-10-15。实现提交：`7c297e8`。状态：本项实现与合同验收通过，暂停在 TODO-01 之前。本次补充说明只修改文档。

返回[主计划](../../p2o_agent_implementation_plan.md)；原始命令和结果见[验收记录](../test_records/TODO-00.md)，容器与初始化见[环境文档](../environment.md#recovery-evidence)。

## 1. 这一步解决了什么

TODO-00 给后续模块定义共同的数据格式、接口和边界校验。之后的环境适配器、collector、评分器、GEPA 和 trainer 可以用同一套任务身份、轨迹结果、token 记录及历史状态交换数据，尽早拒绝串任务、串实例、错误评分或不完整证据。

**现在能执行的是记录构造、字段/语义校验、序列化恢复、已规范化轨迹的结果汇总，以及模拟环境测试。**真实环境接入、模型生成、CRM loss、困难判定/调度、模板搜索、训练、checkpoint 文件读写与 GPU 资源切换都还没实现。不能通过新增的模块启动 P2O 训练。

## 2. 文件在哪里，先读什么

下表路径相对仓库根目录，链接可直接进入源码；在文件中搜索列出的类名/函数名即可定位。

| 文件 | 核心入口 | 已实现内容 |
| --- | --- | --- |
| [verl/trainer/p2o/contracts.py](../../../verl/trainer/p2o/contracts.py) | `Record`、`SampleKey`、`TransitionBatch`、`outcome_from_trace`、`TokenScores`、`P2OState` | 数据记录、版本化序列化和可执行校验；当前业务主体 |
| [agent_system/p2o/base.py](../../../agent_system/p2o/base.py) | `P2OEnvAdapter`、`PromptContextBuilder`、`PolicyGenerator`、`PolicyScorer`、`EpisodeEvaluator` | 公共 Protocol，声明方法签名和职责，方法体尚无真实实现 |
| [tests/trainer/p2o/test_contracts.py](../../../tests/trainer/p2o/test_contracts.py) | `FakeAdapter`、`run_fake_episode`、`ContractTests` | 共同测试流程、模拟适配器、16 项测试及多个子场景 |
| [tests/trainer/p2o/fixtures/traces.json](../../../tests/trainer/p2o/fixtures/traces.json) | 两条 `synthetic` 轨迹 | 按原生 worker 输出结构构造的 ALFWorld/WebShop 模拟输入 |
| [tests/trainer/p2o/export_examples.py](../../../tests/trainer/p2o/export_examples.py) | 模块执行入口 | 导出可审阅、可复现的完整合同样例 |
| [TODO-00-examples.json](../test_records/TODO-00-examples.json) | `cases` 及 token/评分样例 | 已保存的完整输入/输出；不是实机运行结果 |

三个包的 `__init__.py` 位于 `verl/trainer/p2o/`、`agent_system/p2o/`、`tests/trainer/p2o/`。本项没有修改原训练入口/trainer、原环境/collector/reward/actor/core_algos 或两个 init 脚本。

## 3. 核心逻辑：顺着一次数据流看

### 3.1 先区分“任务”和“这次初始化出来的实例”

`SampleKey(env, split, task_catalog_version, task_id)` 是任务身份，使用带转义的 JSON 数组生成规范字符串。seed、价格、worker/slot 不进入这个身份。例如同一个 WebShop goal，在两次合法初始化中可以有不同 `instance_id`，但仍是同一个难样本。

`TaskRecipe` 保存任务选择器、数据/代码版本、构造配置、允许变化字段等轻量材料；`recipe_id` 的计算不依赖原失败实例摘要。`EvalInstance` 单独保存具体初始化材料、实例身份和操作前 t=0 观察，必要时引用世界状态 artifact。这样后续既能按任务重新采样，也能对已经选定的评估实例进行 replay。

本项只实现这些记录及校验。真实 goal/gamefile 的稳定映射、允许改变什么、是否真的有随机多样性，以及如何重建世界，留给 TODO-02。

### 3.2 reset/step 明确指定操作哪些槽位

接口声明采用 `reset_instances(tuple[ResetRequest, ...])` 和 `step(tuple[ActionRequest, ...])`。每条请求/响应携带 slot、request、attempt、episode、instance、policy 等身份。`ResetBatch.assert_requests` / `TransitionBatch.assert_requests` 按 slot 和完整请求身份校验回包，不依赖回包顺序。

例如只向 slot 1、3 发请求，响应先返回 3 再返回 1 可以通过；少一条、多出 slot 2、混入旧实例结果或执行了另一动作都会报错。`ActionRequest` 还会拒绝 padding 行，防止模型补齐数据触发真实环境动作。

这些检查已经执行并测试；真实环境的 inactive slot 不推进、RNG 不变化、救援 subset 隔离仍需 TODO-02/11B 实机验证。当前 fake adapter 测试不能证明真实 worker 已满足这些条件。

### 3.3 把轨迹汇总为统一结果，success 不从 reward 猜

目前跑通的模拟测试流程是：

```text
fixtures/traces.json
  → FakeAdapter.reset_instances：返回初始观察与可用动作
  → run_fake_episode：逐轮选 fixture 提供的动作并调用 step
  → TransitionBatch.assert_requests：检查回包身份/动作
  → EpisodeTrace：检查同一episode、连续turn和终止边界
  → outcome_from_trace：汇总为 EpisodeOutcome
```

`outcome_from_trace` 的输入已经是 adapter 规范化的数据。它读取最后一条显式 `success`，汇总 `raw_reward` 为 `task_return`，并单独累计非法动作惩罚。真实原生字段的提取还没接入。

正常结束/截断需要明确的 success 和终止前快照，避免读到 auto-reset 后的新 episode；尚未结束的轨迹不能汇总。基础设施错误使用 `valid=False, success=None, termination_reason="infra_error"`，不会被补成任务失败。因此“执行失败”“任务未成功”“成功但有惩罚”可以分别表示。

### 3.4 保存原始生成 token，保证两种上下文评分指向同一回答

`PromptSnapshot` 保存消息、token ids、mask、position ids 和 tokenizer 身份；`ContextPair` 保存 base/augmented 两份上下文。无模板时要求两份相同。

`GeneratedResponse` 保存模型原始 response token ids、mask 和逐 token AUG log-prob。`TokenScores.assert_response` 检查 row、policy、prompt hash、response hash 和 mask 是否对应，拒绝另一条回答、另一模型版本或重新分词后的动作文本冒充原始生成结果。

B 版后续的数据流约定为：rollout 直接提供 AUG，BASE 只评分一次，并供 CRM/PPO 复用。当前只落实记录、接口和对齐检查；尚无真实 tokenizer/生成/评分调用，也没有实现 CRM mask 或 PPO loss。本项 token/log-prob 数值都是手工测试数据，包含 EOS 与 pad id 相同的情况。

### 3.5 历史记录必须有最终入选模板的实际成功证据

`GroupEvidence` 保存组内逐轨迹结果，校验重复轨迹和跨组混入；本项允许记录缺 K 条或 invalid 的证据，尚不执行 `≤1/8` 判难。

`TemplateCycle` 的成功证据必须来自本周期最终 `selected` 模板、对应 policy 且 outcome 确实有效成功；候选搜索时成功、最后却落选，不能登记为历史解法。`P2OState` 分开保存 `interval_hard`、`ever_hard`、周期模板登记表和 `latest_solved_cycle`，检查引用和最近成功周期是否一致。

例如 C1 的入选模板解出任务 A，C2 没解出 A，历史仍指向 C1；C3 的入选模板再次解出 A 后才指向 C3。这个检查已经实现；选择/分配模板、计算采样权重、更新这些状态和执行 GEPA 都属于后续 TODO。

### 3.6 可序列化不等于已实现断点续训

所有记录经 `Record` 校验，使用以下信封格式：

```json
{"schema_version": 1, "record_type": "SampleKey", "payload": {"env": "webshop", "split": "train", "task_catalog_version": "fixture-canonical-goals-v1", "task_id": "500"}}
```

`state_dict` / `from_state_dict` 支持嵌套版本化往返，拒绝未知/缺失字段、错误类型/版本、NaN/Infinity 和非 JSON 对象。构造时复制输入、导出时返回独立快照。dataclass 字段冻结，但内部字典不是深层不可变对象，调用者仍应按只读方式使用。

`ComponentState` 给后续搜索/RNG/sampler/资源状态提供组件自有版本边界，`ArtifactRef` 只保存 URI 与 SHA-256。当前没有 artifact 读写、原子 checkpoint 提交、模型/优化器恢复或崩溃恢复流程；这些要在 TODO-09 等模块验收。

## 4. 测试结果，以及结果能说明什么

以下是 TODO-00 实现时已经执行并记录的结果；本次仅补文档，没有重新运行代码测试。完整命令、输出摘要和样例指纹见[验收记录](../test_records/TODO-00.md)。

| 环境 | Python | 合同测试 | 测试体耗时 |
| --- | --- | --- | --- |
| `verl_zhangkaiqi2024_alfworld` | 3.12.11 | 16 tests，0 failures/errors，OK | 0.059s |
| `verl_zhangkaiqi2024_webshop` | 3.10.12 | 16 tests，0 failures/errors，OK | 0.071s |

两容器导出的完整 JSON 样例逐字节一致。16 项测试覆盖 schema/版本/JSON 往返、输入与导出快照隔离、任务与实例身份区分、同任务不同实例不能跨 train/dev、显式能力检查、subset/reorder/padding、终止/截断/infra/惩罚、原始 token 对齐与空输出、历史引用和落选候选证据等。

两条模拟轨迹的公共输出为：

| 模拟输入 | success | task_return | task_score | steps | valid |
| --- | --- | --- | --- | --- | --- |
| ALFWorld | true | 10.0 | null | 1 | true |
| WebShop | false | 0.0 | 0.4 | 2 | true |

这说明同一测试流程能处理两种任务的规范化结果，也说明部分 task_score 不等于成功；它们不是实际 benchmark 成绩。没有运行真实环境 episode、加载模型权重/tokenizer、生成回答、验证真实 log-prob 数值、训练 update 或测试 checkpoint/资源切换。

## 5. 当前问题与已解决问题

| 项目 | 当前状态及影响 |
| --- | --- |
| 宿主机 user-site 覆盖 PyTorch（P2O-ENV-001） | 后续使用用户指定容器；新容器未复现该问题，不再是当前阻塞 |
| 两个旧容器无法访问 `/ceph_home`（P2O-ENV-002） | 已按授权删除重建并分别执行 init 脚本，挂载恢复；旧挂载失效的底层原因未确认 |
| WebShop 附加 pip 源 TLS 重试 | 最终初始化命令临时忽略该附加源后 exit 0；初始化脚本和包版本约束未改 |
| TODO-00 本项 | 已验收，无已知未解决的阻塞问题；实机环境/模型能力仍未验证 |
| 后续语义选择 | 合法随机变体、dev 比例 25% 提案、GEPA 选择器/预算、全零分数与空 selected 的退化采样等，按[待决表 §14.3](../decisions.md#section-14-3)到对应模块讨论 |

两容器已完成 GPU 0–7 的小规模 CUDA 算术检查；这只证明基础计算可用，不证明 vLLM/FSDP 或 8→7+1→8 切换已可用。容器/镜像、挂载、模型路径、初始化命令、版本和日志位置集中在[环境文档](../environment.md)。

## 6. 当前停点与下一步

只完成 TODO-00，其余 14 项仍未执行。下一项 TODO-01 将建立独立入口、trainer、配置和四个任务/策略脚本，验证配置解析与 baseline 隔离；不提前实现 GEPA/CRM，具体门槛见[TODO 清单](../todo_list.md)。等待用户反馈后再开始。

以后每次大 TODO 完成或遇到需讨论的问题，都必须同时更新模块说明/验收记录和在对话中汇报：实现内容、关键文件与核心逻辑、测试命令与结果、证据范围、问题/待决项及下一步。不能仅回复“完成，见文档”，也不能只写 Markdown 而不解释。
