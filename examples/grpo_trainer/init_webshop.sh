set -e

export http_proxy=http://124.16.138.148:7890
export https_proxy=http://124.16.138.148:7890
export all_proxy=socks5://124.16.138.148:7890
export HTTP_PROXY=http://124.16.138.148:7890
export HTTPS_PROXY=http://124.16.138.148:7890
export ALL_PROXY=socks5://124.16.138.148:7890
export NO_PROXY_LIST=localhost,127.0.0.1,127.0.1.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,.local,.lan,.internal,.cluster.local,kubernetes.default.svc
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple


cd /ceph_home/zhangkaiqi2024/verl-agent

# Pyserini loads Lucene through pyjnius when WebShop workers start.
# The base image has a Java runtime but no javac, which pyjnius requires.
if ! command -v javac >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-11-jdk-headless
fi
javac -version

python3 -m pip install \
  'gym==0.24.0' 'Flask==2.1.2' 'Werkzeug==2.1.0' \
  'cleantext==1.1.4' 'rank-bm25==0.2.2' \
  'pyserini==0.17.0' 'faiss-cpu==1.7.4' 'selenium==4.2.0' 'thefuzz==0.19.0' \
  'tensordict==0.8.3' 'datasets==4.0.0'


python -m pip install --upgrade "wandb==0.22.3"

python3 -m spacy download en_core_web_sm
python3 -m pip install -e . --no-deps

python3 - <<'PY'
import sys

import datasets
import spacy

sys.path.insert(0, "agent_system/environments/env_package/webshop/webshop")
import verl.trainer.main_ppo
from web_agent_site.envs import WebAgentTextEnv

spacy.load("en_core_web_sm")
for split in ("train", "test"):
    path = f"data/webshop/text/{split}.parquet"
    dataset = datasets.load_dataset("parquet", data_files=path)["train"]
    assert len(dataset) > 0, f"Empty WebShop dataset: {path}"
    print(f"{split}: {len(dataset)} rows")
print("WebShop imports and datasets OK")
PY
