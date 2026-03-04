# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"

# Pre-compile regex (avoid re-compilation per call)
_WHITESPACE_TAG_RE = re.compile(r"\s*(<|>|/)\s*")

def format_reward(response: str) -> float:
    if "<think>" in response:
        return 0.0

    think_end = response.find("</think>")
    if think_end == -1:
        return 0.0

    after_think = response[think_end + len("</think>"):]
    if "\\boxed{" in after_think:
        return 1.0

    return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def length_penalty(response_length: int, accuracy: float, max_response_length: int = 8192) -> float:
    if accuracy > 0:
        return 0.0
    return -min(response_length / max_response_length, 1.0)

def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = _WHITESPACE_TAG_RE.sub(r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        len_penalty = length_penalty(reward_input["response_length"], accuracy_score)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + 0.1 * len_penalty,
                "format": format_score,
                "accuracy": accuracy_score,
                "length_penalty": len_penalty,
            }
        )

    return scores
