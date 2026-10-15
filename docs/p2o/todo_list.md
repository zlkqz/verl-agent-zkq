# P2O：串行TODO与逐项验收门槛

返回[主计划入口](../p2o_agent_implementation_plan.md)。正文保留原章节编号；跨文件的 `§` 引用按主入口的[章节导航](../p2o_agent_implementation_plan.md#章节导航)定位。设计与待做项不代表已实现，当前实现范围见[TODO-00说明](modules/TODO-00.md)。

<a id="section-9"></a>

## 9. 严格串行 TODO LIST（TODO-00已验收，等待反馈进入01）

**公共门槛：每一项都包含实现+测试+数据核验+文档更新+向用户汇报并停止。不得跳过上一项的门槛直接展开下一大模块，也不同时开多个大TODO。**用户发出继续指示后再进入下一项。遇到问题/测试失败或不确定，按§0.1停下来问用户；无GPU时不得把mock通过标为实机通过。后续有单机8卡A100和真实Qwen3-8B，可以做有界功能实验，但不启动正式长训练。

### [x] TODO-00：公共合同模块与验收样例

- **当前状态：**已实现schema v1合同、公共Protocol、两环境模拟trace、16项测试与完整JSON输入输出；两个任务容器初始化完成后均测试通过，跨Python版本样例一致。详见[TODO-00验收记录](test_records/TODO-00.md)。本项完成后暂停，未进入TODO-01。
- **实现范围：**contracts/types、sample_key/task_id/instance身份与历史状态合同、EnvAdapter协议、TaskRecipe/EvalInstance/Transition/EpisodeOutcome/ContextPair/TokenScores/P2OState，字段schema及版本；定义optional capability与异常类型，包括subset reset/step、terminal snapshot、原始token/logprob传递能力。先阅读两env真实输出，不先写整个trainer。
- **输入→预期输出：**模拟一条ALFWorld和一条WebShop trace，均得到相同结构的EpisodeOutcome；缺失success/instance identity返回明确错误，不能补0。
- **测试/数据核验：**序列化roundtrip；合法/非法schema；公共evaluator可用fake adapter执行而不识别任务名；train/dev sample_key隔离fixture（同key不同随机实例不得跨边）；所有后续模块的数据入口/出口能在合同中定位。
- **通过门槛与停点：**两任务差异仅在adapter能力说明，API示意变成明确合同；记录尚需落实的合法随机化字段/退化采样边界（不重新询问已确认身份与两策略）并汇报，等待继续。

### [ ] TODO-01：独立入口与配置模块

- **实现范围：**复制并改名main/trainer；新Hydra配置、新脚本和模型路径；此时只建P2O扩展位，不实现GEPA/CRM业务。
- **输入→预期输出：**四个新脚本dry-run解析到`两个独立P2O入口 → 各自对应trainer`；原脚本仍指向`main_ppo → RayPPOTrainer`。两任务grouping分别trajectory/step，默认reward/penalty/采样参数与原脚本对齐。
- **测试/数据核验：**imports/配置compose/dry-run；原入口/trainer/脚本git差异为空；新入口P2O关闭时在固定小batch与旧路径比输入/奖励/advantage；只修改必要参数以使用正确模型并记录。检查auto-resume目录隔离和绝对模型路径存在。
- **通过门槛与停点：**脚本不能落回旧trainer，baseline无P2O依赖；提交解析配置对照和文件清单，汇报后停止。

### [ ] TODO-02：环境适配与初始化模块（两任务都过关）

- **实现范围：**公共factory及ALFWorld/WebShop adapters；capture_task、make_instances、reset_instances、episode_outcome；限定为§5的初始化和结果合同，不接优化器。
- **输入→预期输出：**一份任务配方产生全新t=0实例；replay同面板返回相同任务条件；resample变化只在许可字段，得到新instance_id。8条训练组的核心任务与初始化一致；新变体不继承原success。
- **测试/数据核验：**真实两环境reset→固定合法action前缀→reset；不同worker/seed/重复session/auto-reset；只reset/step救援subset且其他slot状态/RNG不变；WebShop价格与合法约束一致、shuffle后仍映射同核心任务/原split；ALFWorld实际多样性测量；异常goal/gamefile拒绝；内存/history清零。比较旧GRPO环境路径固定seed行为不变。
- **通过门槛与停点：**交付小量真实输入/输出样例及任务/实例/可见差异表；未确定随机化边界先问用户，不伪造变体多样性；两任务均通过后汇报停止。

### [ ] TODO-03：每轮上下文与长度模块

- **实现范围：**P2O collector接ContextPair/episode template；新入口自动4096+2048，幂等resume配置；每轮注入，base快照保存和字段传递。
- **输入→预期输出：**≥5轮轨迹中每轮恰好有一份模板、history保持原最近2步；去模板等于当时base输入；空模板与baseline有效token相同；第2次加载配置仍6144。
- **测试/数据核验：**WebShop无history降级分支仍有模板；新episode不继承上个模板；2048边界/超长模板/4096基础观察/response上限；token ids、attention/position/EOS、DataProto选择/padding/balance一致；原始response token与AUG逐token对齐，不能用decode后重新tokenize的动作计算CRM；真实Qwen tokenizer以及一次接近有效上限的GPU生成，检查engine最大长度≥所需总长度。
- **通过门槛与停点：**输出逐turn模板计数、历史步号、base/aug/response长度及去模板diff，原GRPO仍4096；汇报停止。

### [ ] TODO-04：困难缓存与周期调度模块

- **实现范围：**K组success采集、interval_hard/ever_hard/latest_solved_cycle分离、去重/落盘、每epoch检查间隔≥20且H≥32的GEPA调度器；GEPA此时用fake callback，不占用反思模型。
- **输入→预期输出：**0/8、1/8入池，2/8不入；成功轨迹有惩罚仍算成功；未收齐8条不分类；epoch20 H=30、21 H=31不触发，22 H=32触发；完成后最早42再触发，不因GEPA额外增加global_step。
- **测试/数据核验：**不同轨迹长度、重复padding/重试/迟到回包、同任务不同instance/policy、区间归档清空且历史永不因成功/TTL淘汰、H=31等待/H=32拆24+8/H=80拆60+20/H=240拆190+50；历史H大但本区间不足32仍等待；同任务不同seed/价格/worker仍命中、不同goal/gamefile不误命中；epoch不足20即使H足够也等待；正常无改进完成重置锚点、崩溃不重置；resume保持锚点、跨epoch积累、终止预算前不做无用GEPA；原训练sampler调用数和RNG消费不受fake搜索影响；过滤之前捕获0/8难组、同轮重复sample_key不同group分别判定，padding不新增样本。
- **通过门槛与停点：**给出success原始向量与分类结果、触发事件时间线、缓存统计；更新默认参数提案并汇报停止。

### [ ] TODO-05：B版 CRM 与 loss 模块

- **实现范围：**§2的B路径；base全批一次、aug直接取rollout log-prob，同一L_base复用；增强actor重算只在独立验证脚本中作参考；共享actor/core_algos可选CRM，保留旧默认行为。此时用固定手写模板，不需要GEPA。
- **输入→预期输出：**手工logprob矩阵产生预期token mask；PPO分母严格为base；相同response在两上下文打分；CRM=0只消除对应PG贡献，原loss reduction不变。
- **测试/数据核验：**上下界/NaN/全mask/正负advantage、无模板行；response含think/action/EOS和截断，action解析修改不得改变评分token；step和trajectory两种GRPO；实际多轮batch评分、padding/balance/不同microbatch；调用计数证明不重复算base且生产路径增强评分调用为0；使用真实模型做固定batch的前向、反向及最多几次受控update，检查梯度/损失与参考实现和旧路径一致。
- **通过门槛与停点：**保存有效token对齐、logprob/CRM分位数、loss/gradient差异和时间/显存；不能用训练reward看起来正常代替数值验证；汇报停止。

### [ ] TODO-06：Feedback 与模板规则模块

- **实现范围：**B型关键窗口、可选C型对照、reflection prompt、模板解析/长度/泄漏检测；adapter负责任务特定证据和规则context，此阶段不启动反思LLM。
- **输入→预期输出：**两任务失败/成功/超时/非法动作trace得到有step证据且token有界的FeedbackRecord；硬编码ASIN/固定动作解法被拒，通用搜索/检查选项规则被接受。
- **测试/数据核验：**长页面/50步/重复动作裁剪、关键目标保留、无反馈字段不杜撰、隐藏答案不进入prompt、2048边界、注入内容、误拒/漏检fixtures。人工展示两任务各一份反思输入和接受/拒绝模板样例给用户审阅。
- **通过门槛与停点：**规则原因可解释、无LLM审查依赖；模板不能要求SimpleMemory之外的未提供记忆；记录反馈策略和规则版本，汇报停止。

### [ ] TODO-07：独立冻结推理 Worker 模块

- **实现范围：**纯推理P2OInferenceWorker、model identity、generate/drain/close；独立1卡πinit组及policy-eval组创建接口，不做训练组动态切换。
- **输入→预期输出：**πinit从`/ceph_home/arknet/hf_models/Qwen/Qwen3-8B`生成合法模板候选；policy-eval加载指定当前快照；不创建optimizer/额外FSDP权重副本。
- **测试/数据核验：**1卡真实反思、两组GPU id不重叠、模型fingerprint正确、decode/tokenizer一致、输入过长/解析失败可返回明确状态；生成后drain/close资源释放，父进程退出不遗留子进程。记录A100容量、dtype、reflection输入输出长度及并发，不能只测试一条很短prompt就宣布长上下文安全。
- **通过门槛与停点：**有真实反思输出通过TODO-06规则，显存与进程生命周期证据完整；确定有界token预算并汇报停止。

### [ ] TODO-08：GEPA 搜索模块（冻结policy独立运行）

- **实现范围：**Evaluator、面板、Pareto/接受/覆盖/分配、预算；使用已完成的adapter/反馈/推理组，以固定当前policy做一次独立搜索，不接完整训练外循环。
- **输入→预期输出：**已知score向量fixture得到正确Pareto及覆盖集合；一小组真实TaskRecipe→固定面板→规则通过候选→评价→模板集，跨候选面板完全相同。
- **测试/数据核验：**两任务分别跑；空hard、H不足、全部失败、无改进、全部规则拒绝、预算不足、重试、tail slots；对比实际调用与预算账本；检查train/dev sample_key无交集、dev固定环境和greedy解码重复/重启评分一致、dev轨迹不进reflection、新实例不冒充新任务、policy全程不变、不增加训练update。
- **通过门槛与停点：**真实GEPA一轮完整、终止路径齐全；给用户报告候选/反馈/拒绝与接受例子及消耗；需要改变score/覆盖/随机化策略时先讨论；通过后汇报停止。

### [ ] TODO-09：checkpoint 与断点恢复模块

- **实现范围：**§8持久化、原子提交、artifact保留、Math阶段标记的agent适配；序列化已有各模块状态，不先把全训练循环接起来才补救。
- **输入→预期输出：**保存/加载interval_hard、ever_hard、latest_solved_cycle、历轮模板/权重、触发cursor、面板、候选/预算/RNG完全一致；GEPA_COMMITTED恢复不重复搜索；未完成request不会被重复计入score。
- **测试/数据核验：**对TRAIN_SAVED、GEPA_RUNNING、GEPA_COMMITTED等点故障注入；重启后检查模型/optimizer/scheduler、candidate文本与面板一致；损坏/缺shard拒绝并保留旧checkpoint；keep_latest不删pin；6144不累加；核验环境sampler恢复能力和uncertain request成本。不能只做pickle roundtrip。
- **通过门槛与停点：**实际进程退出/重启的恢复证据通过，列出仍不支持的恢复边界；汇报停止。

### [ ] TODO-10：GPU 阶段生命周期模块

- **实现范围：**资源状态机，排空→checkpoint→关闭N卡组→N−1+1推理→提交→关闭→恢复N卡，使用已验证checkpoint模块。
- **输入→预期输出：**目标8卡完成8→7+1→8；GEPA前后训练参数/optimizer状态连续，反思始终πinit；训练恢复后一次固定batch update与未切换参考匹配容差。
- **测试/数据核验：**可先2→1+1→2定位，再必须做8卡；检查actor/PG/GPU/pid、模型identity、worker归一化配置不二次除DP、scheduler/RNG、超时/异常清理、不影响其他run、切换期间无未完成generation/reward future。
- **通过门槛与停点：**正常及故障恢复都验证，报告切换时间/I/O/显存；不得把7卡静态训练当完成归还需求；汇报停止。

### [ ] TODO-11A：historical策略模块及两任务集成

- **实现范围：**仅historical新trainer接历史命中、最新求解周期选择/最新dev回退、GEPA生命周期；先完成这一策略并汇报停下，再做11B。
- **输入→预期输出：**样本在cycle1解出、cycle2没解出仍选1；cycle3再次解出改选3；最终selected中的同模板两个已评变体1成功1失败也记为解出；落选模板即使成功也不建专属引用，未评价不得记解出；只曾判难未解出取最新dev；未曾判难不预加模板。同任务不同seed/价格命中，不同任务标识不误命中。
- **测试/数据核验：**两个任务分别有限update跨至少两次GEPA；验证interval池清空而历史不丢、旧模板不被清理、采样权重/RNG及resume一致、逐turn模板与CRM正确、原随机sampler/grouping不变。可用小阈值smoke但另测真实20/32调度fixture。
- **通过门槛与停点：**给出sample_key→历史证据→实际模板的可追踪记录；checkpoint恢复与故障恢复通过；更新文档并汇报，等待继续。

### [ ] TODO-11B：failure_retry策略模块及两任务集成

- **实现范围：**独立failure_retry trainer控制首次无模板K组→全失败→初始环境恢复→增强K组；复用公共评分/GEPA/选择器。首次组不参与更新、任一已评变体成功即算解出均已确认。
- **输入→预期输出：**0/8触发一次，1/8不触发但入难池，invalid不冒充失败；未曾判难也可救援；无模板可用不救援；重试失败不循环；同sample_key、同instance与policy，不继承终止状态。
- **测试/数据核验：**两个真实环境初始prompt/world价格/goal/seed一致、memory清空；base/rescue request去重及数据选择；最终batch由未救援base组+救援rescue组构成，先替换完整组再算reward/advantage，首次失败行不得混入；不能拼成2K作同一GRPO组；epoch/globalstep不因救援额外增加；中断在base之后/rescue之后/update之后恢复无重复更新。首次失败有非法动作惩罚的非零advantage例子必须展示，以明确舍弃或保留语义。
- **通过门槛与停点：**两任务短程运行通过，分别报告base/rescue成功率、episode/token/时延、更新组数与模板来源。汇报并暂停，不开始正式训练。

### [ ] TODO-12：B版性能与数值专项验收

- **实现范围：**对已实现B做分桶/避免冗余entropy及数据传输优化；A仅在离线小batch作参照，不增加生产增强评分。
- **输入→预期输出：**相同冻结数据的AUG、CRM、loss/gradient在事先讨论容差内；生产调用计数增强评分=0、BASE不重复。
- **测试/数据核验：**真实0.8/0.95解码、长短轨迹、最大模板、token边界、policy/context/response版本一致；两个策略分别测含救援的总成本，不只测评分；如概率口径不一致，停问用户，不能静默回退A。
- **通过门槛与停点：**记录省去的增强评分时间、整体时间/显存及误差；不得仅凭功能测试宣称不掉点；汇报停止。

### [ ] TODO-13：回归与实现交付模块

- **实现范围：**汇总接口/脚本/配置/恢复说明与已完成TODO证据，整理正式实验尚待执行事项，不增加新算法功能。
- **输入→预期输出：**新旧入口清晰隔离；共享代码在CRM=None/不开P2O时保留原损失/环境行为；接手用户可按文档运行有界验证与恢复。
- **测试/数据核验：**原GRPO配置/导入/关键reward/advantage/loss回归，四个新脚本dry-run，模型和产物路径、状态版本、no-template验证、数据split、资源清理；复核代码变动与记录一一对应，未测项明确列出。
- **通过门槛与停点：**最终报告实现了什么、哪些真实8卡测试已过、哪些仅mock、哪些仍未验证、正式实验建议；等待用户是否启动正式训练的指令。

<a id="section-10"></a>

## 10. 后续正式实验（不属于上述自动执行范围）

保留现有随机训练采样协议，与两个指定baseline分别比较相同初始模型、训练update/episode预算、解码/奖励/grouping/penalty和评估任务。P2O新增的搜索生成、反思、评分、切换和策略2失败救援成本均单独计量；episode数相同不等于GPU成本相同。正式实验需用户另行安排，不在完成TODO后自行启动。

主指标是无模板成功率；WebShop同时报告原始task_score，ALFWorld按任务类型分组。记录两策略模板命中率、首次/救援成功率、ever_hard/interval大小、样本/核心任务独立计数、新变体成功率、零task方差组、CRM过滤率、规则拒绝率、搜索开销。建议之后比较GRPO/完整P2O/去CRM/去蒸馏与预算接近GRPO；随机变体、泄漏规则和CRM快速路径如果改变默认方法，应独立说明并做相应验证。

评估集/seed/任务身份在成对对照中固定，不从GEPA-dev选择最终成绩；用于选checkpoint的集合明确标dev。ALFWorld验证固定任务可由独立评估adapter实现，不要求改变原训练采样。多seed和完整训练后的表现才支持“不掉点/泛化收益”结论，功能smoke不能替代。
