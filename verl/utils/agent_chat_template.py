"""Chat template options shared by dataset and agent rollout tokenization."""


REASONING_TAG_INSTRUCTION = "This reasoning process MUST be enclosed within <think> </think> tags."


def get_chat_template_kwargs(data_config):
    """Omit a null enable_thinking so the model template retains its default."""
    kwargs = dict(data_config.get("apply_chat_template_kwargs", {}))
    if kwargs.get("enable_thinking") is None:
        kwargs.pop("enable_thinking", None)
    return kwargs


def apply_generation_chat_template(tokenizer, messages, data_config):
    """Render the model template before the input is encoded for vLLM."""
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **get_chat_template_kwargs(data_config),
    )
