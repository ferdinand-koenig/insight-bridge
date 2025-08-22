# test_hetzner_send.py
import asyncio
import json
import os
from pathlib import Path
import yaml

from app.llm_worker_provisioners.hetzner import HetznerProvisioner

CONFIG_PATH = Path("config.yaml")
backend_file = Path("cache/backend.json")

with backend_file.open() as f:
    backend = json.load(f)

print("Loaded backend:", backend)


async def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    hetzner_cfg = cfg["hetzner"]

    key= os.getenv("HETZNER_API_TOKEN")

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

    await provisioner.terminate_backend(backend)






if __name__ == "__main__":
    asyncio.run(main())
