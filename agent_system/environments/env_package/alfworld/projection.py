# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def alfworld_projection(actions, action_pools, return_diagnostics=False):
    """Parse a single action and check the PRE-action admissible commands.

    Commands are supplied by ALFWorld rather than a duplicated command grammar.
    Thinking tags and reasoning language are diagnostics, not validity rules.
    Invalid responses yield an empty command instead of executing a text suffix.
    """
    if len(action_pools) != len(actions):
        raise ValueError("One admissible-actions entry is required per response")

    parsed, valids, diagnostics = [], [], []
    for text, pool in zip(actions, action_pools):
        blocks = re.findall(r"<action>(.*?)</action>", text, flags=re.I | re.S)
        format_valid = (len(blocks) == 1
                        and len(re.findall(r"</?action\b[^>]*>", text, re.I)) == 2)
        action = blocks[0].strip().lower() if format_valid else ""
        # Syntax only checks that the payload is one nonempty command line.
        # The environment's command pool is authoritative for command semantics.
        syntax_valid = bool(action) and not re.search(r"[\x00-\x1f\x7f<>]", action)
        canonical_commands = {command.strip().lower(): command for command in (pool or ())}
        executable = bool(syntax_valid and action in canonical_commands)
        valid = bool(format_valid and syntax_valid and executable)

        parsed.append(canonical_commands[action] if valid else "")
        valids.append(valid)
        diagnostics.append({
            "action_format_valid": bool(format_valid),
            "action_syntax_valid": bool(syntax_valid),
            "action_executable": executable,
            "think_tags_complete": bool(re.search(r"<think>.*?</think>", text, re.S)),
            "think_open_count": text.count("<think>"),
            "think_close_count": text.count("</think>"),
            "response_has_cjk": bool(re.search(r"[\u4e00-\u9fff]", text)),
        })

    if return_diagnostics:
        return parsed, valids, diagnostics
    return parsed, valids
