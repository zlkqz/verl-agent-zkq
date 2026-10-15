# ALFWorld / WebShop 的两个独立开关

两个主脚本直接接受以下环境变量，不再使用模型类别或中间模式映射：

| 变量 | 默认值 | 行为 |
| --- | --- | --- |
| `PROMPT_REQUIRE_THINK` | `false` | `true` 时在每轮任务 prompt 添加 `This reasoning process MUST be enclosed within <think> </think> tags.`；`false` 时不添加 |
| `ENABLE_THINKING` | `null` | `null` 时不向 `apply_chat_template` 传 `enable_thinking`，使用模型模板默认行为；`true` / `false` 时直接传对应布尔值 |

两个开关完全独立，六种组合均可使用。任务 prompt 始终要求先分析当前情况，再输出 `<action>...</action>`。

```bash
# 默认：不要求 think 标签，使用模型模板默认值。
bash my_scripts/run_alfworld.sh

# 普通 instruct 模型：要求推理使用 think 标签，不传原生 thinking 开关。
PROMPT_REQUIRE_THINK=true ENABLE_THINKING=null \
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct EXP_NAME=my_instruct_experiment \
  bash my_scripts/run_alfworld.sh

# 直接关闭模板 thinking，不要求模型输出 think 标签。
PROMPT_REQUIRE_THINK=false ENABLE_THINKING=false EXP_NAME=my_non_thinking_experiment \
  bash my_scripts/run_webshop.sh
```

Qwen3-8B 非 thinking 专用脚本仍可直接运行，分别在对应任务容器中执行：

```bash
bash my_scripts/run_alfworld_non_think.sh
bash my_scripts/run_webshop_non_think.sh
```

这两个包装脚本直接设置 `PROMPT_REQUIRE_THINK=false`、`ENABLE_THINKING=false`。Qwen3 模板会在 assistant 输入末尾生成空 `<think>\n\n</think>\n\n`，随后模型生成标签外的推理和 action。空 block 由模型模板提供，代码不手动拼接，也不对其他模型模板强加这个格式。

模型路径可通过 `MODEL_PATH` 覆盖。可用 `DRY_RUN=1` 只打印命令和保存路径，不启动训练。

## 实验名

主脚本保留原来的 `qwen3_8b_think` 默认命名，非 thinking 包装脚本直接写明自己的 `qwen3_8b_non_think` 默认名称。修改开关不会自动修改名称，没有 `reasoning_name` 或 `model_name` 变量。切换模型、开关或超参数时，实验名由使用者自行指定。

主脚本及包装脚本均使用 `EXP_NAME=${EXP_NAME:-默认名称}`，可从外部覆盖：

```bash
export EXP_NAME=my_custom_experiment
bash my_scripts/run_alfworld_non_think.sh
```

未设置或为空时使用默认值。已有消融脚本的 `EXP_NAME_SUFFIX` 只参与主脚本默认命名；显式指定 `EXP_NAME` 后原样使用，不再追加后缀。保存目录和日志文件名随完整实验名更新。

## 核心配置

```yaml
data:
  prompt_require_think: false
  apply_chat_template_kwargs:
    enable_thinking: null
```

数据加载、超长 prompt 过滤和多轮 rollout 共用模板参数处理。调用 `apply_chat_template` 前，只将值为 `null` 的 `enable_thinking` 移除，其余模板参数原样保留。vLLM 收到的是已经构造好的 `prompt_token_ids`。

两个任务的 invalid action 判定均不依赖 think 标签。ALFWorld 检查唯一的 action block 及当前 admissible commands；WebShop 检查唯一的 action block、语法及执行前页面的 click 目标。think 标签信息仅作为诊断记录，空输入 block 不会拼入模型 response。

## 验证

```bash
TEST_QWEN3_TOKENIZER_PATH=/ceph_home/arknet/hf_models/Qwen/Qwen3-8B \
TEST_INSTRUCT_TOKENIZER_PATH=/ceph_home/arknet/hf_models/Qwen/Qwen2.5-7B-Instruct \
  python3 -m pytest -q tests/trainer/test_agent_reasoning_modes.py
```

覆盖全部六种开关组合、实际 tokenizer / collector / dataset / 过滤边界，以及实验名与开关相互独立的行为。未设置本地 tokenizer 路径时跳过对应集成用例，不加载模型权重。
