"""Print reviewable synthetic inputs/outputs: python3 -m tests.trainer.p2o.export_examples."""

import json

from tests.trainer.p2o.test_contracts import FIXTURES, ContractTests, FakeAdapter, instance, recipe, run_fake_episode
from verl.trainer.p2o.contracts import SCHEMA_VERSION, TokenScores


def examples():
    cases = []
    for i, fixture in enumerate(FIXTURES):
        task = recipe(i)
        adapter = FakeAdapter(fixture)
        trace, result = run_fake_episode(adapter, task)
        cases.append({"fixture_kind": fixture["fixture_kind"], "sample_key": task.sample_key.canonical,
                      "recipe": task.state_dict(), "instance": instance(task).state_dict(),
                      "trace": trace.state_dict(), "outcome": result.state_dict()})
        adapter.close()
    response = ContractTests().response()
    scores = TokenScores(response.row_id, response.generation_policy_version, "base", response.contexts.base.prompt_hash,
                         response.response_hash, response.response_mask, (-1, -2, -3, 0))
    scores.assert_response(response)
    return {"schema_version": SCHEMA_VERSION, "real_environment_execution": False,
            "note": "Synthetic contract fixtures, not environment/model/GEPA results.",
            "cases": cases, "raw_response_example": response.state_dict(), "base_score_example": scores.state_dict()}


if __name__ == "__main__":
    print(json.dumps(examples(), ensure_ascii=False, indent=2, allow_nan=False))
