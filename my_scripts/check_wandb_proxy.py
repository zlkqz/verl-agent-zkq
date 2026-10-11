"""Check W&B connectivity from a Ray worker using the training runtime env."""

import argparse
import os

import ray
import requests

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@ray.remote
def probe(test_wandb_init):
    proxy_env = {
        key: os.environ.get(key)
        for key in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "no_proxy",
            "NO_PROXY",
        )
    }
    result = {"worker_proxy_env": proxy_env}
    try:
        response = requests.post(
            "https://api.wandb.ai/graphql",
            json={"query": "{ __typename }"},
            timeout=15,
        )
        result.update(status=response.status_code, body=response.text[:200])
    except Exception as exc:
        result["error"] = repr(exc)

    if test_wandb_init:
        try:
            import wandb

            with wandb.init(
                project="p2o_agent_exp",
                name="proxy-diagnostic",
                job_type="proxy-diagnostic",
                settings=wandb.Settings(init_timeout=60),
            ) as run:
                result["wandb_run_url"] = run.url
        except Exception as exc:
            result["wandb_init_error"] = repr(exc)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wandb-init",
        action="store_true",
        help="Create and finish a small online W&B diagnostic run",
    )
    args = parser.parse_args()
    ray.init(runtime_env=get_ppo_ray_runtime_env())
    print(ray.get(probe.remote(args.wandb_init), timeout=120 if args.wandb_init else 30))
