# ALFWorld 与 WebShop 的 GRPO 训练调用链

本文按当前仓库的 `examples/grpo_trainer/run_alfworld.sh` 和 `run_webshop.sh` 追踪代码。这里的“AgentRuntime”指实际负责模型与环境往返的实现：`TrajectoryCollector`、`EnvironmentManager` 和各环境的 Ray worker。仓库在这条调用链中没有名为 `AgentRuntime` 的独立服务。

## 1. 从脚本到训练循环：共用链路

```text
run_{alfworld,webshop}.sh
  ├─ python -m examples.data_preprocess.prepare --mode text
  │    └─ 分别写 data/{alfworld,webshop}/text/{train,test}.parquet
  └─ python -m verl.trainer.main_ppo (Hydra 参数覆盖)
       ├─ ray.init → TaskRunner.remote().run(config)
       ├─ make_envs(config) → 训练/验证 EnvironmentManager + Ray 环境 actors
       ├─ 加载 Qwen tokenizer/processor、数据集、EpisodeRewardManager
       ├─ TrajectoryCollector(config, tokenizer, processor)
       └─ RayPPOTrainer.init_workers() → fit()
            ├─ 可选验证 _validate() → TrajectoryCollector.multi_turn_loop(..., is_train=False)
            └─ 每个训练 batch → multi_turn_loop(..., is_train=True)
                 → episode reward → 旧策略/ref log-prob → GRPO advantage → actor update
```

入口见 [`examples/grpo_trainer/run_alfworld.sh`](../examples/grpo_trainer/run_alfworld.sh)、[`run_webshop.sh`](../examples/grpo_trainer/run_webshop.sh)、[`prepare.py`](../examples/data_preprocess/prepare.py)、[`main_ppo.py`](../verl/trainer/main_ppo.py) 和 [`ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py)。预处理脚本虽然读取 Geometry3K 来取得指定数量的行，却把题目内容丢掉，生成空文本 prompt；真正的任务在环境 `reset()` 时产生。parquet 在此主要确定模态及 DataLoader 大小，不是 ALFWorld/WebShop 的训练题库。

两个脚本均以 Qwen2.5-1.5B-Instruct、vLLM（可用首个脚本参数改 `ENGINE`）、2 GPU、16 个训练样本、128 个验证样本、`env.rollout.n=8` 启动。训练时 prompt 被按组连续复制 8 次，故一个 batch 有 **16×8=128 条独立环境轨迹**；验证组数为 1，有 128 条环境轨迹。两脚本差异还包括最大 prompt 长度（2048/4096）、最大环境步数（50/15）、actor mini/micro batch（256/32 与 64/8）、rollout/ref log-prob micro batch（32 与 16）、项目名。这些是资源和长度参数差异，训练算法链路相同。脚本末尾的 `$@` 可追加 Hydra 覆盖；首个位置参数同时会成为 `ENGINE`，因此使用时需留意参数位置。

`main_ppo.py` 选择 FSDP worker（默认配置）及 episode reward manager，并断言 `actor_rollout_ref.rollout.n == 1`：这里的 GRPO 采样组由 **`env.rollout.n`** 实现，并非 verl 常见的单轮模型输出重复参数。`RayPPOTrainer.fit()` 调用 [`TrajectoryCollector.multi_turn_loop`](../agent_system/multi_turn_rollout/rollout_loop.py)；默认 `algorithm.filter_groups.enable=False`，因此进入 `vanilla_multi_turn_loop`，不进入动态采样分支。验证同样走环境循环，但不复制为 8 组。`trainer.val_before_train=True` 会先验证，此后按 `trainer.test_freq=5` 验证。

## 2. 一步 Agent 与环境交互的实际执行

[`rollout_loop.py`](../agent_system/multi_turn_rollout/rollout_loop.py) 的 `vanilla_multi_turn_loop` 执行如下：

1. `envs.reset()` 返回 `obs={'text', 'image', 'anchor'}` 和 `infos`。文本 manager 将原始观察、任务、可用动作和最多 2 步历史（配置 `env.history_length=2`）拼成模型要读的 user 消息。
2. `preprocess_batch()` 对每条观察应用 tokenizer 的 chat template，生成 `input_ids`、mask 等；当前两脚本都是纯文本。环境观察每步重新构造 prompt，历史由 `SimpleMemory` 保存，不是向模型服务维护一条持久聊天会话。
3. 把 batch 填充到 rollout worker 数可整除，调用 `actor_rollout_wg.generate_sequences()` 生成本步响应，再去掉填充；`batch_decode()` 得到每条文本动作。
4. `EnvironmentManager.step()` 用环境专属 `projection` 提取 `<action>...</action>`，检查是否含 `<think>...</think>` 和中文字符，设置 `is_action_valid`，再调用环境向量包装器的 `step(actions)`。动作格式不合法时会把响应尾部文本交给环境；有效性标志由解析规则决定，不等同于检查动作是否真的在可用动作集合中。
5. 保存当前 prompt、response、环境 reward、有效标志与 `active_masks`；累计每条轨迹的 episode reward 和长度。已结束轨迹仍会占据批量位置，但后续步的 `active_masks=False`，不会进入最终有效训练样本。全体结束或达到 `env.max_steps` 时停止。
6. `gather_rollout_data()` 仅收集 active 的决策步；同一条轨迹的每个有效决策步都附上相同的累计 episode reward。`uid` 标识同一个 8 条采样组，`traj_uid` 标识单独轨迹。两者是 GRPO 分组所需的关键字段。

奖励路径见 [`EpisodeRewardManager`](../agent_system/reward_manager/episode.py) 和 [`ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py)：每个有效决策步的 response 最后一个有效 token 放入该轨迹的累计 episode reward；不按长度归一。随后启用的 invalid action penalty 对无效格式动作在该 token 减 `0.1`。脚本设置 `algorithm.use_kl_in_reward=False`，因此不在 reward 上扣 KL；但 actor 的 `use_kl_loss=True`，会在策略损失中使用参考策略的 KL，系数 `0.01`。接着 `compute_advantage(..., grpo)` 调用 [`compute_grpo_outcome_advantage`](../verl/trainer/ppo/core_algos.py)：按 `uid` 将同任务的样本归组，把 response reward 求和为标量，在组内减均值、默认除标准差，再把 advantage 广播到该步 response 的有效 token。当前实现默认 `compute_mean_std_cross_steps=True`，均值/标准差统计包含同组各轨迹的**各个有效步骤**；这意味着较长轨迹在统计里有更多条目，不能简单理解成“每条轨迹只计一次”。最后计算旧策略/ref log-prob、更新 actor；GRPO 不使用 critic value 更新。

## 3. “异步”究竟在哪一层

这里有三种容易混淆的含义：

| 层 | 实际行为 |
| --- | --- |
| 环境 worker 并发 | ALFWorld 和 WebShop 各自为每个环境实例建一个 Ray actor。向量环境的 `reset`/`step` 对所有 actor 先发 `.remote()`，再统一 `ray.get(futures)`。各环境可同时运行，但**每个环境步有整批等待屏障**，慢环境会拖住下一轮模型生成。 |
| ALFWorld 内部异步 | `AlfredTWEnv.init_env()` 调用 TextWorld `register_games(..., asynchronous=True, batch_size=1)`；这是 TextWorld 封装的实现设置，外层仍等待 Ray 整批返回。 |
| verl 的 async rollout | 基础配置 `actor_rollout_ref.rollout.mode: sync`，脚本没有覆盖。`ray_trainer.fit()` 中普通 async rollout manager 的调用被注释，实际直接调用 `TrajectoryCollector.multi_turn_loop()`。所以这两个示例**没有启用持续流水线式异步 agent rollout**；训练 batch 仍是“采一批轨迹 → 算优势 → 更新模型”。 |

Ray `TaskRunner.remote()` 和 GPU worker group 也是远程 actor 调用；这提供进程/设备分布式执行，并不意味着 agent 每条轨迹独立异步推进。配置里的 `reward_model.launch_reward_fn_async` 默认也为 `False`。需要估算并发资源时，训练 128 个环境 actor 加验证 128 个环境 actor 都会被创建，脚本每个 actor 请求 `0.1` CPU；WebShop 工厂还按 actor 数量做短暂 sleep 等待启动。

## 4. 两种环境专属交互

### ALFWorld：TextWorld 文本游戏

`make_envs()` 识别 `alfworld/AlfredTWEnv`，通过 [`config_tw.yaml`](../agent_system/environments/env_package/alfworld/configs/config_tw.yaml) 指定 TextWorld 游戏数据路径 `$ALFWORLD_DATA`、训练集和验证集，创建 [`AlfworldEnvs`](../agent_system/environments/env_package/alfworld/envs.py) 与 [`AlfWorldEnvironmentManager`](../agent_system/environments/env_manager.py)。每个 Ray `AlfworldWorker` 持有一个 `base_env.init_env(batch_size=1)` 游戏环境。相邻 8 个 worker 的 seed 相同，形成 GRPO 组；reset 时具体游戏仍由 ALFWorld/TextWorld 环境选择。验证使用 seed+1000、组数 1。

`reset()` 返回文字观察和 `admissible_commands`；manager 从开场文本提取 `Your task is to: ...`，将观察和允许的命令（过滤 `help`）放进 [`ALFWORLD_TEMPLATE`](../agent_system/environments/prompts/alfworld.py)。模型给出含 `<think>`、`<action>` 的文本，projection 提取动作字符串；worker 用 `self.env.step([action])` 操作 TextWorld。新的观察、`won`、`admissible_commands` 回到 manager，进入下一次 prompt。文本模式下环境 reward 是 `10 * won`，成功率由最后一个 active step 的 `info['won']` 计算；任务类型还可由 gamefile 汇总。这里选择的是 TW 文本环境，不是 THOR 图像仿真环境。

### WebShop：模拟网站购物任务

`make_envs()` 创建 [`WebshopMultiProcessEnv`](../agent_system/environments/env_package/webshop/envs.py) 与 [`WebshopEnvironmentManager`](../agent_system/environments/env_manager.py)。默认 `env.webshop.use_small=True`，读取仓库内 1000 商品版本的 JSON；`human_goals=False`。每个 `WebshopWorker` 通过 Gym 创建自己的 [`WebAgentTextEnv`](../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py)。训练从 goals 下标 `[500, len(goals))` 选任务，验证从 `[0,500)` 选；一次 reset 抽 16 个不同 goal 下标，各重复 8 次，确保同一 GRPO 组共享购物目标。worker 的随机种子也按组相同。训练与验证 worker 分别用 `seed` 与 `seed+1000` 洗牌目标列表，所以上述下标范围不能保证实际目标互不重叠。

原始观察的 `[SEP] Instruction: [SEP] ...` 给出任务；manager 提取 instruction，将页面文本、可用操作和历史拼进 [`WEBSHOP_TEMPLATE`](../agent_system/environments/prompts/webshop.py)。可用动作由环境的 `has_search_bar` 与 `clickables` 变成 `search[<your query>]`、`click[文字]`。模型响应经 projection 后交给 `WebAgentTextEnv.step(action)`；其解析 `search[...]`/`click[...]`，由 `SimBrowser.search/click` 直接调用同进程 `SimServer.receive()` 更新页面。无效动作在站点模拟器里通常保持状态且 reward 为 0。worker 保存原始 `task_score`，仅在 `done and reward == 1.0` 时改为训练 reward 10、`won=True`，其余训练 reward 为 0。环境内部在完成后可能自动 reset，轨迹采集器以返回的 `done` 停止计入该轨迹。

WebShop 代码里虽然构造了 `http://127.0.0.1:3000` 风格 URL，但此路径上的 `SimBrowser` 直接调 Python `SimServer.receive()`；这两个脚本**不要求另起网站 HTTP 服务**。它也不启动真实浏览器。WebShop 的 `task_score` 只作为额外评估指标，训练用上述二值成功 reward。

## 5. Docker、沙盒与 Harbor

这两个脚本及其实际环境调用链没有 Docker 创建/销毁 API，也没有 Harbor SDK/服务调用。环境隔离边界是 **Ray actor 进程**：各 actor 各持一个 Python 环境对象；Ray 负责调度和资源声明，不负责为每条轨迹建容器。ALFWorld 需要预备其游戏数据与 TextWorld 依赖，WebShop 需要商品/属性 JSON、Lucene 索引与其 Python 依赖，模型训练需要 verl/vLLM 等环境。可以为了安装和复现把整个训练进程放进 Docker 镜像，但这是部署选择，不是本示例 Agent 与环境交互的运行前提。

仓库其他任务包含 `SandboxFusion` 代码执行评分功能；这两个脚本采用 `EpisodeRewardManager` 的环境奖励，不调用代码沙盒。WebShop 页面操作和 ALFWorld 文本命令也不执行模型生成的任意 Python/ shell 代码，因此这条实现链路无需 Harbor 一类的容器管理器来管理逐任务沙盒。

## 6. 当前 `data/` 的数据量与任务样例

需要区分 **DataLoader 的 Parquet 行数**、**原始任务池大小** 和 **每步采样量**。当前两个脚本均把预处理结果写在各自的 `data/<任务>/text/` 下。Parquet 由 [`prepare.py`](../examples/data_preprocess/prepare.py) 按 `--train_data_size 16 --val_data_size 128` 生成，记录的 prompt 是空文本；任务在环境 `reset()` 时从真实任务池取得。

| 任务 | 处理好的 `train.parquet` / `test.parquet` | 实际任务池 train / 默认 test | 原始数据及筛选口径 |
| --- | ---: | ---: | --- |
| ALFWorld | 16 / 128 行 | **3,553 / 140** 个可加载游戏 | `json_2.1.1/train` 有 6,374 条 `traj_data.json`，其中 3,553 条通过 [`collect_game_files()`](../agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py) 的任务类型、非 movable/切片、游戏文件存在及 `solvable` 筛选；`valid_seen` 有 251 条原始轨迹、140 个可加载游戏。另有 `valid_unseen`：255 条原始轨迹、134 个可加载游戏，需改 `env.alfworld.eval_dataset` 才会用到。 |
| WebShop | 16 / 128 行 | **6,410 / 500** 个目标下标 | [`items_shuffle_1000.json`](../data/webshop/items_shuffle_1000.json) 有 1,000 件商品；结合 [`items_ins_v2_1000.json`](../data/webshop/items_ins_v2_1000.json) 的合成指令与选项可生成 6,910 个 goal。列表下标 0–499 给验证、500–6909 给训练。一件商品可以产生多个目标。 |

ALFWorld 的可加载数按当前环境代码条件逐文件统计；WebShop 目标数按 `load_products()` 与 `get_synthetic_goals()` 的选项组合规则计算。WebShop 各 worker 在切分下标前按各自 seed 洗牌列表，训练与验证 seed 不同，因此“6,410 / 500”只表示下标范围大小，**不保证两边的目标内容无交集**。每个训练迭代只从任务池抽 16 个任务，各复制成 8 条环境轨迹；验证抽 128 条，跨迭代会重新采样。

### 各看一条真实任务

**ALFWorld**：一条训练 [`traj_data.json`](../data/alfworld/json_2.1.1/train/pick_two_obj_and_place-TissueBox-None-SideTable-321/trial_T20190908_025939_647835/traj_data.json) 的 `task_type=pick_two_obj_and_place`，目标物体是 `TissueBox`，目标容器是 `SideTable`。人工描述为 “Put two boxes of tissues on the nightstand.”。同目录的 [`game.tw-pddl`](../data/alfworld/json_2.1.1/train/pick_two_obj_and_place-TissueBox-None-SideTable-321/trial_T20190908_025939_647835/game.tw-pddl) 包含这个场景的世界状态、目标条件与可执行文字命令；其中 `walkthrough` 示范了找两盒纸巾并放到边桌。训练 agent 只收到 `reset/step` 的观察和可用动作，不直接读取人工轨迹或 walkthrough。

`walkthrough` 可以看作这局游戏的一条参考解法，用于理解游戏或离线检查。当前 GRPO 训练和验证**不把它当监督答案，也不按它比对动作**：游戏筛选只检查同一文件里的 `solvable` 标志；TextWorld 注册请求 `won`、`admissible_commands`、`gamefile`，未请求专家动作；`AlfworldWorker` 取得的 TextWorld 原始 `scores` 也未作为训练奖励，实际奖励是 `10 * info['won']`。所以模型可以走不同于 walkthrough 的路线，只要达到游戏的目标状态，同样算成功。仓库另有可选的 `AlfredExpert`/`is_solvable()` 代码用于专家或可解性工作，但此配置中 `use_expert=False`，不会把专家计划送入这条 GRPO 交互链路。

**WebShop**：商品 `B08XQRDDT1` 是 YOUNG VISION Cream Blush Stick。属性文件给它的合成指令为 “Find me long lasting, easy apply, cruelty free lips makeup”；商品有 `color: set-a / set-b` 两种选择，因此形成两个候选购物目标。agent 用 `search[查询词]` 找商品、`click[商品/选项/Buy Now]` 选择购买，最后根据商品、属性和选项与 goal 的匹配得到分数。它只是可生成目标的样例；一次 `reset()` 是否抽中它由随机采样决定。

## 7. `logic` 与 `search_engine` 在 AgentRuntime 中的调用

### ALFWorld 的 `logic/`

[`config_tw.yaml`](../agent_system/environments/env_package/alfworld/configs/config_tw.yaml) 将 `logic.domain` 指向 [`alfred.pddl`](../data/alfworld/logic/alfred.pddl)，将 `logic.grammar` 指向 [`alfred.twl2`](../data/alfworld/logic/alfred.twl2)。前者定义世界类型、谓词及动作规则，后者定义文字命令与反馈语法。它们是生成游戏的基础素材；当前每个 `game.tw-pddl` 已内嵌 `pddl_domain`、`grammar`、该场景的 `pddl_problem`、`solvable` 和 `walkthrough`。游戏文件内嵌的内容可能经过任务定制，不能假定与两个基础 logic 文件逐字相同。

实际训练调用是 `make_envs()` → `AlfworldEnvs` → `AlfredTWEnv.collect_game_files()` 筛选 `game.tw-pddl` → `textworld.gym.register_games()` → `AlfworldWorker.reset/step()` → TextWorld 根据游戏逻辑返回观察、`admissible_commands`、`won`。[`AlfredTWEnv.get_game_logic()`](../agent_system/environments/env_package/alfworld/alfworld/agents/environment/alfred_tw_env.py) 虽然会直接打开 `logic.domain/grammar`，但这条训练路径没有调用它。因此已生成游戏的实时交互直接依赖 `game.tw-pddl` 内嵌逻辑；`data/alfworld/logic/` 主要用于生成或重新构建游戏。当前脚本设置 `ALFWORLD_DATA=data/alfworld` 的绝对路径，用于展开游戏数据目录。

### WebShop 的 `search_engine/`

默认 `use_small=True` 选 1,000 商品 JSON，但 `WebAgentTextEnv` 的 `num_products=None`，所以 [`init_search_engine()`](../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py) 选择 `indexes` 目录创建 Pyserini `LuceneSearcher`。仓库中该目录通过符号链接指向 [`data/webshop/search_engine/indexes`](../data/webshop/search_engine/indexes)。商品 JSON、属性 JSON 和本地 Lucene 索引都需可读。

初始化链：`WebshopWorker.__init__` → `gym.make(WebAgentTextEnv-v0)` → `WebAgentTextEnv.__init__` → `SimServer.__init__` → `load_products()` 加载商品和属性，`init_search_engine()` 打开索引。模型发出 `search[关键词]` 时：`WebAgentTextEnv.step()` → `SimBrowser.search()` → `SimServer.receive()` → `search_results()` → `get_top_n_product_from_keywords()` → `LuceneSearcher.search(keywords, k=50)`。命中的 Lucene 文档先映射回 ASIN，再过滤掉不在当前商品 JSON 中的商品；结果每页显示 10 件，并被渲染为下一步文本观察。普通查询走 Lucene；`<r>`、`<a>`、`<c>`、`<q>` 等特殊前缀走代码里的专用分支。这里是本地搜索索引调用，无需独立 HTTP 搜索服务或 Docker 容器。

## 8. 一条真实数据如何走完 `Env → Action → 新 Env → Observation`

本节选第 6 节中的真实游戏和商品。**文件里的 walkthrough、商品和指令是真实数据；下文的模型回复与示意观察是按源码推导的交互示例，并非实测训练日志。** 当前命令行环境没有安装 TextWorld、Ray、Gym、Pyserini，因此不能把某次随机 reset 的完整文字输出伪装成固定事实。

### 8.1 ALFWorld：把两盒纸巾放到边桌

真实 [`game.tw-pddl`](../data/alfworld/json_2.1.1/train/pick_two_obj_and_place-TissueBox-None-SideTable-321/trial_T20190908_025939_647835/game.tw-pddl) 的 `walkthrough` 是：

```text
go to diningtable 1 → take tissuebox 3 from diningtable 1
→ go to sidetable 1 → move tissuebox 3 to sidetable 1
→ go to diningtable 1 → take tissuebox 2 from diningtable 1
→ go to sidetable 1 → move tissuebox 2 to sidetable 1
```

**Env 里有什么？** TextWorld 持有该游戏的 `pddl_domain` 动作规则、`grammar` 文字规则、`pddl_problem` 场景及目标，以及当前可变的世界状态。可变状态用谓词表达，包括 agent 当前地点 `atLocation`、物品在容器中 `inReceptacle`、手持物 `holds` 等。这个游戏的目标要求**两个不同的 TissueBox 都在 SideTable**。每个 `AlfworldWorker` 持有一个 TextWorld Gym 实例；外层 [`AlfWorldEnvironmentManager`](../agent_system/environments/env_manager.py) 还保存任务文字、最近观察和动作历史。模型看不到内部谓词或 walkthrough。

**Observation 与 Action 有什么？** `reset()` 通过 `AlfworldEnvs.reset()` 并发调用 worker 的 `self.env.reset()`，得到当前地点的文字反馈与 `info['admissible_commands']`。manager 将它们放进 [`ALFWORLD_TEMPLATE_NO_HIS`](../agent_system/environments/prompts/alfworld.py)，向模型展示“当前观察 + 当前可用的文字命令 + 输出格式”。后续提示还含最近两步历史。动作是自然语言游戏命令，例如真实 walkthrough 中的 `go to diningtable 1`、`take tissuebox 3 from diningtable 1`、`move tissuebox 3 to sidetable 1`，模型须把它放在 `<action>...</action>` 内，并有 `<think>...</think>`。

以第一步为例，若 `go to diningtable 1` 在当时可用命令中，示意模型响应为：

```text
<think>I need to reach the tissue boxes.</think><action>go to diningtable 1</action>
```

[`alfworld_projection`](../agent_system/environments/env_package/alfworld/projection.py) 提取动作字符串；`AlfWorldEnvironmentManager.step()` → `AlfworldEnvs.step()` → 对应 worker 的 `self.env.step([action])`。该游戏的 `GotoLocation` PDDL 规则要求 agent 在起点且目的地有 receptacle；成功后删除旧 `atLocation`，加入新 `atLocation`。TextWorld 再按更新后的世界状态生成**新地点反馈**及**新的 admissible commands**。manager 把前一步观察与动作写入历史，将新反馈、可用命令和历史拼成下一条模型输入。具体反馈句子与命令清单取决于 reset 后的真实状态，不能从 walkthrough 唯一确定。

沿真实 walkthrough 继续：`take tissuebox 3 from diningtable 1` 对应 `PickupObject`，需要纸巾盒在桌上且手为空；成功后清除 `inReceptacle(tissuebox 3, diningtable)`，加入 `holds(agent, tissuebox 3)`。`move tissuebox 3 to sidetable 1` 对应 `PutObject`，成功后加入 `inReceptacle(tissuebox 3, sidetable)`，并清除手持状态。第二盒也放到边桌后，目标谓词成立，`won=True`、`done=True`；外层将 `won` 转为奖励 10。未成功的普通步骤奖励为 0。`is_action_valid` 只检查模型响应格式；动作能否改变世界由 TextWorld 判定。

```text
模型可见：地点反馈 + admissible commands + 历史
       ↓ <action>go to diningtable 1</action>
projection → TextWorld step → 更新 atLocation
       ↓
新地点反馈 + 新 admissible commands → manager 拼下一步 observation
```

### 8.2 WebShop：搜索、选颜色、购买商品

真实商品 `B08XQRDDT1` 的基础指令是 “Find me long lasting, easy apply, cruelty free lips makeup”，选项为 `color=set-a/set-b`。[`get_synthetic_goals()`](../agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py) 因而为它生成两个颜色目标，并可能附加价格上限。一次 `reset()` 会从洗牌后的 goal 列表抽取目标；这里以 `set-a` 分支说明，不声称该商品一定被抽中。

**Env 里有什么？** `SimServer` 持有商品/属性目录、Lucene 索引、目标列表及 `user_sessions`。一个 session 保存隐藏的 `goal`、当前 `keywords`、页码 `page`、商品 `asin`、所选 `options`、访问/购买计数、`done` 和 `reward`。`SimBrowser` 保存 session ID、当前 URL 和 HTML；`WebAgentTextEnv` 从 HTML 产出文本观察并解析当前可点控件。外层 [`WebshopEnvironmentManager`](../agent_system/environments/env_manager.py) 另存最近两步历史。隐藏 goal 的 ASIN 和选项结构不会直接作为字段发给模型；模型看到自然语言购买要求和当前页面。

**Observation 与 Action 有什么？** `reset()` 建立 session，清空 `keywords/page/asin`、设置 `options={}`，渲染带搜索框的首页。[`get_available_actions()`](../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py) 从 HTML 找搜索框、按钮、商品链接和颜色等 radio 选项；manager 将搜索框表示成 `search[<your query>]`，将可点控件表示成 `click[文字]`。首页给模型的 observation 是“购物目标 + 首页文字 + 可用搜索动作”。示意回复：

```text
<think>Search for the requested makeup.</think><action>search[long lasting cruelty free lips makeup]</action>
```

[`webshop_projection`](../agent_system/environments/env_package/webshop/projection.py) 取出 `search[...]`；`WebshopMultiProcessEnv.step()` 将它交给对应 worker。`WebAgentTextEnv.step()` 解析关键词，调用 `SimBrowser.search()` → `SimServer.receive()` → `search_results()`。session 随即更新 `keywords`、`page=1`、`asin=None`、`options={}`；Lucene 检索前 50 个命中，过滤到当前商品目录、每页显示 10 件，生成结果页 HTML。`WebAgentTextEnv.observation` 将可见 HTML 转成 `[SEP]` 分隔的文字，worker 还返回新页的 `has_search_bar/clickables`；manager 把目标、页面文字、可用 `click[...]` 和历史拼成下一次模型输入。**该商品是否出现在示例查询的第一页取决于实际索引排序**。

如果结果页显示该商品且它可点，接下来可依次执行 `click[B08XQRDDT1]`、`click[set-a]`、`click[Buy Now]`。点商品使 session 的 `asin` 改为该 ASIN、HTML 变成商品页；点颜色使 `options['color']='set-a'` 并重新渲染商品页；点 Buy Now 调用 [`SimServer.done()`](../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py)，按所买商品、目标、价格和选项算原始分数，设置 `done=True`，渲染完成页。`WebshopWorker.step()` 仅在 `done and reward == 1.0` 时给训练奖励 10，否则给 0；原始分数保留为 `task_score`。环境自身在完成后会 reset 新 session，但轨迹采集器依据已返回的 `done` 结束旧轨迹，因此终止步返回的可用动作即便属于新 session，也不会继续用于旧轨迹。

```text
目标 + 首页 → search[关键词] → 更新 keywords/page → Lucene → 结果页及可点 ASIN
           → click[ASIN]   → 更新 session.asin     → 商品页及颜色选项
           → click[set-a]  → 更新 session.options  → 更新后的商品页
           → click[Buy Now]→ 计算分数，done=True    → 完成页、终止奖励
```

这里 `is_action_valid` 仍只检查输出格式。`click[...]` 是否真的可执行由当页 `text_to_clickable` 决定；不在其中的 click 进入无效动作分支，页面通常不变、reward 为 0。两个任务都把**新 observation 重新包装成一条 user prompt**，再进入下一次 `generate_sequences()`。

## 9. 模型实际收到的命令类型和完整调用形状

### ALFWorld：`admissible_commands` 具体有哪些

`info['admissible_commands']` 是 **TextWorld 根据当前游戏状态枚举的完整命令字符串列表**。它不是一份每步固定的动词表；同一个游戏在不同地点、手持不同物品时会给不同的实例化命令。以第 8 节真实游戏内嵌的 grammar 为准，可能出现的命令模板如下：

| 类别 | 文本命令模板 | 例子/条件 |
| --- | --- | --- |
| 观察 | `look`、`inventory`、`examine {receptacle/object}` | 看房间、背包或对象 |
| 移动 | `go to {receptacle}` | `go to diningtable 1` |
| 容器 | `open {receptacle}`、`close {receptacle}` | 目标容器可开关时 |
| 拾取 | `take {object} from {receptacle}` | `take tissuebox 3 from diningtable 1`；在场且手为空等条件成立 |
| 放置 | `move {object} to {receptacle}`、`put {object} into {outer object}`、`put {outer object} in {receptacle}` | `move tissuebox 3 to sidetable 1`；必须持有且目标可容纳 |
| 使用/加工 | `use {object}`、`heat {object} with {receptacle}`、`clean {object} with {receptacle}`、`cool {object} with {receptacle}`、`slice {object} with {sharp object}` | 不同任务和状态才会出现；`use` 对应 ToggleObject |
| 帮助 | `help` | TextWorld 支持，但 manager 在加入模型 prompt 前过滤掉 |

这些是**游戏语法允许的命令形式**，不是声称任意时刻全都可执行。真实列表由 TextWorld 的 `EnvInfos(admissible_commands=True)` 返回，`AlfWorldEnvironmentManager.build_text_obs()` 只过滤 `help` 并直接格式化列表。`envs.py` 中另定义了 `ALF_ACTION_LIST` 常量，但这条训练链路没有用它来生成提示或限制模型动作。`alfworld_projection()` 也不检查模型输出是否属于当前列表；如果格式符合却选了当下不可执行的命令，TextWorld 决定反馈和状态变化。

### 两个模型调用的完整可读示例

以下是**用真实游戏/商品字段填入代码模板的教学示例**。`current_observation`、ALFWorld 当步命令列表及 WebShop 搜索排序属于运行时动态值，示例将它们构造成一个合理情境，**不是从本机实测抓取的原始日志**。真正送给 Qwen tokenizer 时，`TrajectoryCollector.preprocess_single_sample()` 将整块文本作为一条 `user` 消息应用 chat template，并追加 assistant 生成起始标记，形如 `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant`。每一步都是新构造的一条 user prompt。

**调用 A：ALFWorld 真实纸巾盒游戏，假设当前位于餐桌旁。** 下列 `user` 文本按 [`ALFWORLD_TEMPLATE`](../agent_system/environments/prompts/alfworld.py) 填充；`task_description` 来自真实游戏 grammar，命令形式来自同一游戏，具体当步列表是示意。

```text
<|im_start|>user
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: find two tissuebox and put them in sidetable.
Prior to this step, you have already taken 1 step(s). Below are the most recent 1 observations and the corresponding actions you took: [Observation 1: 'Welcome to TextWorld, ALFRED! Your task is to: find two tissuebox and put them in sidetable.', Action 1: 'go to diningtable 1']
You are now at step 2 and your current observation is: You arrive at diningtable 1. You see tissuebox 2 and tissuebox 3.
Your admissible actions of the current situation are: ['look'
 'inventory'
 'go to sidetable 1'
 'take tissuebox 2 from diningtable 1'
 'take tissuebox 3 from diningtable 1'
 'examine diningtable 1'].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
<|im_end|>
<|im_start|>assistant
```

示意模型 **output**：

```text
<think>I need two tissue boxes on the side table; I can pick up one now.</think><action>take tissuebox 3 from diningtable 1</action>
```

`batch_decode` 后交给 `alfworld_projection`，得到实际环境 action `take tissuebox 3 from diningtable 1`，并标记格式有效。TextWorld `step([action])` 若满足前置条件，就把 tissuebox 3 从餐桌移到 agent 手中；下一次 prompt 的 observation 会是**运行时返回的取物反馈 + 新可用命令 + 最近历史**。此时若 agent 仍在餐桌旁，通常需要先 `go to sidetable 1`，到达边桌后才可能出现 `move tissuebox 3 to sidetable 1`。示例中的用户消息展示的是模型看到的文字结构，不能据此断定 reset 后的第一步恰好在餐桌旁。

## 10. TextWorld 如何判定当步的 `admissible_commands`

对于当前训练使用的文本游戏，判定可概括为：**游戏的当前事实集合 `S` + 动作规则的参数类型及前置条件 → 当前可实例化动作 → 文字命令列表**。TextWorld 的 [`GameProgression`](https://github.com/microsoft/TextWorld/blob/main/textworld/generator/game.py) 在 reset 时从初始状态求 `valid_actions`；每次有效 step 先应用动作的 add/remove effects，再重新求 `valid_actions`。[`State.all_applicable_actions`](https://github.com/microsoft/TextWorld/blob/main/textworld/logic/__init__.py) 枚举规则参数的对象绑定，只保留前置条件在当前事实中成立的绑定。随后 [`TextWorldEnv._gather_infos`](https://github.com/microsoft/TextWorld/blob/main/textworld/envs/tw.py) 把动作转成对应文字命令、去重、排序，形成 `admissible_commands`。这是 TextWorld 官方实现的机制说明；当前 ALFWorld 数据的具体规则以各 `game.tw-pddl` 内嵌 PDDL 为准。

用第 8 节真实纸巾盒游戏的规则说明：

| 命令什么时候出现 | 该游戏规则要求的关键事实 |
| --- | --- |
| `go to diningtable 1` | 当前有 agent 所在位置 `atLocation`，目标 diningtable 对应一个 `receptacleAtLocation`；到达后 `atLocation` 改为餐桌位置。 |
| `take tissuebox 3 from diningtable 1` | tissuebox 可拾取、位于 diningtable；agent 在 diningtable 所在位置、手为空；若容器可开，则必须已打开。拿起后删除它的 `inReceptacle`，加入 `holds`/`holdsAny`。 |
| `move tissuebox 3 to sidetable 1` | agent 已手持这盒纸巾、已位于 sidetable 所在位置；目标容器已打开或无需打开，且 `canContain(SideTableType, TissueBoxType)`。成功后纸巾盒进入 sidetable，手变空。 |
| `open` / `close` 某容器 | agent 在该容器所在位置；它必须 `openable`，且分别处于未打开/已打开状态。 |
| `examine` 某物 | 对象在当前所在容器中且可见，或已拿在手里；具体可见性由该游戏规则决定。 |
| `heat` / `clean` / `cool` 某物 | agent 手持相应可加工物，且分别位于 microwave / sink 或 sinkbasin / fridge 所在位置。 |
| `slice` 某物 | 目标可切、agent 在其位置，并手持 Knife 或 ButterKnife。 |

所以“当前有纸巾盒在餐桌上”仍不够让 `take ...` 出现：agent 还必须在餐桌旁、手为空，容器如需打开也得先打开。拿起纸巾盒后，同一盒的 `take ...` 会消失；但 agent 还在餐桌旁时，`move ... to sidetable 1` 也不会因“已经拿着”就立即出现，必须先到边桌。这解释了 walkthrough 的 `take → go to sidetable → move` 顺序。`look`、`inventory` 等观察命令通常持续可用；每步精确列表仍以环境返回的 `info['admissible_commands']` 为准。

模型即使输出列表以外、但标签格式正确的字符串，`alfworld_projection` 仍会把它交给 TextWorld；TextWorld 的 step 根据当前有效命令判定它无效并保持游戏事实不变。模型侧的 `is_action_valid=True` 仅说明输出格式合格，**不代表 TextWorld 动作前置条件成立**。

**调用 B：WebShop 真实商品 `B08XQRDDT1` 的一个合成购物目标。** 这里选 `color=set-a` 目标及一个可生成的价格上限。当前是初始搜索页；首页 HTML 会被转成 `[SEP]` 风格文字，下面把多余空白归并以便阅读。字段排列遵循 [`WEBSHOP_TEMPLATE_NO_HIS`](../agent_system/environments/prompts/webshop.py)。

```text
<|im_start|>user
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: Find me long lasting, easy apply, cruelty free lips makeup with color: set-a, and price lower than 30.00 dollars.
Your current observation is: WebShop [SEP] Instruction: [SEP] Find me long lasting, easy apply, cruelty free lips makeup with color: set-a, and price lower than 30.00 dollars [SEP] Search.
Your admissible actions of the current situation are:
[
'search[<your query>]',
'click[search]',
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
<|im_end|>
<|im_start|>assistant
```

示意模型 **output**：

```text
<think>I should search for the requested makeup before selecting a product.</think><action>search[long lasting cruelty free lips makeup]</action>
```

`webshop_projection` 提取 `search[long lasting cruelty free lips makeup]`；`WebAgentTextEnv.step()` 调 Lucene 搜索并将 session 更新为关键词和第一页结果。下一次 prompt 的 `current_observation` 会是**实际搜索结果页的文字**，`available_actions` 可能包括 `click[某个 ASIN]`、`click[next >]`、`click[back to search]` 等；只有当搜索结果真的包含 `B08XQRDDT1` 时，才能点击它，接着在商品页点 `click[set-a]` 和 `click[buy now]`。注意初始页 HTML 的 Search 按钮也被 `get_available_actions()` 收集为 `click[search]`，但 [`WebAgentTextEnv.step()`](../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py) 明确排除了 `click[search]`；真正应使用的是带查询词的 `search[...]`。这是当前实现中**展示动作与可执行动作不完全一致**的一个具体例子。
