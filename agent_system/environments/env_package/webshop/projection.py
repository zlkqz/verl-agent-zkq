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


def webshop_projection(actions, available_actions=None, return_diagnostics=False):
    """Parse one action and check it against the PRE-action page.

    Reasoning language/think tags are diagnostics, never action validity rules.
    Search is supported from any page by WebAgentTextEnv; click[search] is a no-op.
    Invalid outputs become an empty (no-op) action, never an arbitrary suffix.
    """
    if available_actions is not None and len(available_actions) != len(actions):
        raise ValueError("One available-actions entry is required per response")
    parsed, valids, diagnostics = [], [], []
    for i, text in enumerate(actions):
        blocks = re.findall(r"<action>(.*?)</action>", text, flags=re.I | re.S)
        format_valid = (len(blocks) == 1
                        and len(re.findall(r"</?action\b[^>]*>", text, re.I)) == 2)
        match = re.fullmatch(r"(search|click)\[([^\[\]\r\n]+)\]",
                             blocks[0].strip(), flags=re.I) if format_valid else None
        syntax_valid = match is not None and bool(match.group(2).strip())
        action = ""
        executable = False
        if syntax_valid:
            name, argument = match.group(1).lower(), match.group(2).strip().lower()
            action = f"{name}[{argument}]"
            if name == "search":
                executable = True
            elif available_actions is not None:
                executable = argument != "search" and argument in available_actions[i]["clickables"]
            else:
                # Without page state this API can only validate syntax.
                executable = argument != "search"
        valid = bool(format_valid and syntax_valid and executable)
        parsed.append(action if valid else "")
        valids.append(valid)
        diagnostics.append({
            "action_format_valid": bool(format_valid),
            "action_syntax_valid": bool(syntax_valid),
            "action_executable": bool(executable),
            "think_tags_complete": bool(re.search(r"<think>.*?</think>", text, re.S)),
            "think_open_count": text.count("<think>"),
            "think_close_count": text.count("</think>"),
            "response_has_cjk": bool(re.search(r"[\u4e00-\u9fff]", text)),
        })
    if return_diagnostics:
        return parsed, valids, diagnostics
    return parsed, valids
