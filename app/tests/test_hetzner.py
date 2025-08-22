# test_hetzner.py
import asyncio
import json
import os
from pathlib import Path

import yaml

from app.llm_worker_provisioners.hetzner import HetznerProvisioner

# IMAGE_ID = 105888141  # Docker CE app

CONFIG_PATH = Path("config.yaml")
backend_file = Path("cache/backend.json")
backend_file.parent.mkdir(exist_ok=True)

async def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    hetzner_cfg = cfg["hetzner"]

    key = os.getenv("HETZNER_API_TOKEN")
    if not key:
        raise RuntimeError("HETZNER_API_TOKEN not set in environment")
    provisioner = HetznerProvisioner(
        api_token=key,
        snapshot_name="docker-ce",
        ssh_key_name=hetzner_cfg["ssh_key_name"],
        ssh_private_key_path=os.path.expanduser(hetzner_cfg["ssh_key_path"]),
        private_network_name=hetzner_cfg["private_network_name"],
        server_types=["CX32", "CX42", "CPX31", "CX52", "CPX41"]
    )

    backend = await provisioner.spawn_backend()
    print(backend)
    with backend_file.open("w") as f:
        json.dump(backend, f, indent=2)
    # await provisioner.run_on_backend(backend, prompt="What is an LLM?", timeout=900)


if __name__ == "__main__":
    asyncio.run(main())