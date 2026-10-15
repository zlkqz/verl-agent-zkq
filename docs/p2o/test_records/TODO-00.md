# TODO-00：公共合同与验收样例

日期：2026-10-15；状态：实现与本项验收通过，等待用户反馈后再进入TODO-01。

## 范围与产物

新增 [contracts.py](../../../verl/trainer/p2o/contracts.py) 和 [base.py](../../../agent_system/p2o/base.py)，提供数据记录、边界验证及结构化Protocol。未创建真实环境adapter、P2O入口/trainer、collector、GEPA或CRM实现。

合同模块自身只依赖Python标准库。记录使用 `schema_version=1 / record_type / payload` 信封；嵌套记录逐层版本检查，未知字段、缺失字段、类型错误、非有限数字及非JSON对象均拒绝。构造时复制输入，导出时返回独立快照；dataclass字段冻结，内部字典按只读快照使用。非JSON的张量/RNG/world材料通过带SHA-256的ArtifactRef引用，实际落盘和原子提交属于TODO-09。

| 后续边界 | 本项提供的合同 |
| --- | --- |
| 环境初始化与任务身份（02） | SampleKey、TaskRecipe、EvalInstance、AdapterCapabilities；capture_task/capture_instance/make_instances/reset_instances |
| subset操作与终止结果（02/11B） | ResetRequest/Entry/Batch、ActionRequest、TurnIdentity、Transition/Batch、EpisodeTrace/Outcome；显式slot/request/attempt身份，padding不能产生环境动作，terminal snapshot必填 |
| 每轮上下文（03） | ChatMessage、PromptSnapshot、ContextPair、TemplateBinding；保存base/aug原消息、token ids、mask、position ids、tokenizer身份及hash |
| 原始生成与评分（03/05） | GeneratedResponse、TokenScores、PolicyCapabilities、PolicyGenerator/Scorer；保存原始response ids/mask和逐token AUG，评分检查row/policy/prompt/response/mask，空/错误输出显式标记 |
| 困难记录与历史（04/08/11A） | GroupEvidence、HardSampleRecord、TemplateCycle、SolvingEvidence、LatestSolved、P2OState；区间与ever_hard分开，缺K/invalid证据可记录但本项不分类 |
| 反馈与规则（06） | FeedbackRecord、TemplateRuleContext；可见反馈与verifier私有标识分开，规则实际检测/脱敏在06实现 |
| 推理与GEPA评价（07/08） | PolicyGenerator、EvalRequest/Result、EvaluationPanel、EpisodeEvaluator；显式policy/instance/template/panel/decoding身份，train/dev按SampleKey隔离 |
| 保存与资源状态（07–10） | Record.state_dict/from_state_dict、Stateful.state_dict/load_state_dict、ComponentState、ArtifactRef、P2OState资源phase；组件内部状态schema由对应模块后续细化并版本化 |

SampleKey仅包含 `(env, split, task_catalog_version, task_id)`，规范字符串使用带转义的JSON数组，不把seed/worker/价格加入身份。TaskRecipe的recipe_id由任务身份、selector、版本、构造配置和许可变化规则确定，不依赖原失败实例seed。EvalInstance单独持有初始化材料与t=0观察。

reset/step的公共签名采用显式请求记录：`reset_instances(tuple[ResetRequest, ...])`、`step(tuple[ActionRequest, ...])`。返回顺序可以变化，`assert_requests`按slot及完整请求身份匹配，拒绝缺失、多余、重复或串实例结果。ResetEntry中可经 `request.instance.sample_key.task_id` 取得task_id，不重复存一份可能不一致的任务标识。replay必须显式传入已保存实例；实际重建、slot隔离和RNG恢复由TODO-02实现。

TemplateCycle仅允许最终selected模板的真实成功证据建立历史解法。P2OState检查历史引用完整，并按registry提交顺序验证“最近一次成功周期”；较新周期未覆盖某样本时保留旧成功周期。该检查不选择模板、不规定退化采样、不执行调度/搜索。

## 实际运行与结果

在两个用户指定容器中，完成初始化之后执行：

```bash
docker exec -w /ceph_home/zhangkaiqi2024/verl-agent verl_zhangkaiqi2024_alfworld \
  python3 -m unittest discover -s tests/trainer/p2o -v
docker exec -w /ceph_home/zhangkaiqi2024/verl-agent verl_zhangkaiqi2024_webshop \
  python3 -m unittest discover -s tests/trainer/p2o -v
```

| 容器 | Python | 结果 |
| --- | --- | --- |
| verl_zhangkaiqi2024_alfworld | 3.12.11 | 16 tests，0 failures/errors，OK（0.059s，测试体耗时） |
| verl_zhangkaiqi2024_webshop | 3.10.12 | 16 tests，0 failures/errors，OK（0.071s，测试体耗时） |

测试位于 [test_contracts.py](../../../tests/trainer/p2o/test_contracts.py)，包含嵌套JSON roundtrip、schema/类型/身份错误、输入快照隔离、同任务不同随机实例不跨train/dev、subset/reorder/padding、terminal/截断/infra区分、惩罚不改变success、原始response与logprob对齐（含EOS=pad）、候选落选不能建立解法、旧成功周期保留、新成功周期替代和历史引用完整性。

两条 [synthetic trace fixtures](../../../tests/trainer/p2o/fixtures/traces.json) 依据真实worker源码输出结构构造，经同一fake adapter/evaluator harness得到：

| fixture | success | task_return | task_score | steps | valid |
| --- | --- | --- | --- | --- | --- |
| ALFWorld | true | 10.0 | null | 1 | true |
| WebShop | false | 0.0 | 0.4 | 2 | true |

这里验证的是统一接口与数据语义，**没有运行真实环境episode**。公共harness不判断任务名；具体环境原生字段的提取在TODO-02实现。缺失success/instance身份被拒绝，未补0；infra outcome的success为null，未当作任务失败。

完整输入/输出与原始token/BASE评分例子见 [TODO-00-examples.json](TODO-00-examples.json)，可复现：

```bash
docker exec -w /ceph_home/zhangkaiqi2024/verl-agent verl_zhangkaiqi2024_alfworld \
  python3 -m tests.trainer.p2o.export_examples
```

同一导出命令在WebShop容器执行，输出与已保存JSON逐字节一致（cmp通过）。样例指纹：`fdf491997153698f1d8eed8d5f982d954cd8bff98c404d3cd46f1d5b1c15f9fa`。

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

收尾检查：`git diff --check`通过；所有文档相对链接、代码围栏、已记录源码hash和样例outcome核对通过；计划仅TODO-00勾选，剩余14项未勾选。原入口/trainer、环境/collector/reward/actor/core_algos和两个init脚本的git diff为空。GPU测试进程退出后8卡均恢复占用10 MiB、利用率0%，两个任务容器保持running。

## 保留问题与停点

- TODO-02再验证真实任务目录版本、ALFWorld可产生的初始化多样性、WebShop价格/goal合法变体及split映射；本项未授权或实现新的随机化字段。
- TODO-08/11前确认全零dev分数、空selected、成功子集全零等退化采样行为；当前仅允许表示这些数据，不决定采样算法。
- dev比例25%、选择器与搜索预算等仍按计划§14.3在对应模块讨论。本项未改变已确认的身份、success阈值或两策略。
- 模型权重/tokenizer/config fingerprint实际核验、真实rollout logprob一致性、资源切换与checkpoint I/O均未验证，本项没有用mock替代这些后续验收。
- 原入口、trainer、环境、collector、reward、actor与baseline运行逻辑未修改。`my_scripts/run_webshop.sh`有用户预先存在的一处镜像名注释修改，本次保留且不纳入提交。

下一步为TODO-01（独立入口与配置）。按用户逐大TODO停点要求，本项完成后暂停，等待反馈再开始。
