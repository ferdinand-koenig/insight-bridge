"""
hetzner.py

Provisioner for Hetzner Cloud using SSH for VM lifecycle and HTTP for inference.

Workflow:
1. Spawn VM via Hetzner API using an existing snapshot and SSH key.
2. Start a lightweight HTTP server on the VM exposing /infer.
3. Forward a local port via SSH to access the VM securely.
4. Run inference via HTTP requests.
5. Terminate VM when idle or after use.
"""

import asyncio
from typing import Any, Dict, List, Optional

import requests
from .ssh_tunnel import SSHTunnel
from hcloud import Client
from hcloud.images import Image
from hcloud.server_types import ServerType

from app import logger
from .base_provisioner import BaseProvisioner


class HetznerProvisioner(BaseProvisioner):
    """
    Provisioner for Hetzner Cloud backends.
    """

    def __init__(self, api_token: str, snapshot_name: str, ssh_key_name: str,
                 ssh_private_key_path: str, server_types: Optional[List[str]] = None,
                 private_network_name: str=None):
        """
        Args:
            api_token (str): Hetzner Cloud API token
            snapshot_name (str): Snapshot image name containing preloaded LLM
            ssh_key_name (str): SSH key name registered in Hetzner console
            ssh_private_key_path (str): Local path to private key for SSH
            server_types (List[str]): Ordered list of server types to try.
        """
        self.client = Client(token=api_token)
        self.snapshot_name = snapshot_name
        self.ssh_key_name = ssh_key_name
        self.ssh_private_key_path = ssh_private_key_path
        self.private_network_name = private_network_name
        self.server_types = server_types or ["CX52", "CX42", "CPX41", "CX32", "CPX31"] # default if not provided
        self.locations = ["hel1", "nbg1", "fsn1"]  # TODO make argument

    async def spawn_backend(self, timeout:int=600) -> Dict[str, Any]:
        """
        Spawn a Hetzner Cloud server using the specified snapshot and SSH key.

        Returns:
            dict: Backend information including IP, SSH user, SSH key path, and server ID.
        """
        # TODO add argument for timeout
        # Find snapshot
        snapshots = self.client.images.get_all() # type="snapshot")
        snapshot = next((s for s in snapshots if s.name == self.snapshot_name), None)
        if not snapshot:
            logger.error(f"Snapshot '{self.snapshot_name}' not found")
            raise RuntimeError(f"Snapshot '{self.snapshot_name}' not found")

        # Find SSH key
        ssh_keys = self.client.ssh_keys.get_all()
        key = next((k for k in ssh_keys if k.name == self.ssh_key_name), None)
        if not key:
            logger.error(f"SSH key '{self.ssh_key_name}' not found")
            raise RuntimeError(f"SSH key '{self.ssh_key_name}' not found")

        # Find existing private network
        networks = self.client.networks.get_all()
        net = next((n for n in networks if n.name == self.private_network_name), None)
        if not net:
            logger.error(
                f"Private network '{self.private_network_name}' not found. Create it manually in Hetzner Cloud."
            )
            raise RuntimeError(
                f"Private network '{self.private_network_name}' not found. Create it manually in Hetzner Cloud."
            )

            # Try all server_type + location combinations
        for stype_priority, stype_name in enumerate(self.server_types, start=1):
            stype = ServerType(name=stype_name.lower())
            for loc_priority, loc_name in enumerate(self.locations, start=1):
                try:
                    location = self.client.locations.get_by_name(loc_name)
                    server_name = f"llm-{stype_name.lower()}-{loc_name}-{int(asyncio.get_event_loop().time())}"
                    response = self.client.servers.create(
                        name=server_name,
                        server_type=stype,
                        image=Image(id=snapshot.id),
                        ssh_keys=[key],
                        networks=[net],
                        location=location,
                        public_net=False,  # TODO some bug here
                    )
                    server = response.server
                    server_id = server.id

                    # --- WAIT UNTIL SERVER IS RUNNING ---
                    elapsed= 0
                    interval= 5
                    while elapsed < timeout:
                        s = self.client.servers.get_by_id(server_id)
                        if s.status == "running":
                            break
                        logger.info(f"Waiting for server {server_name} to become active (status: {s.status})...")
                        await asyncio.sleep(interval)
                        elapsed += interval
                    else:
                        logger.error(f"Server {server_name} did not become active within {timeout} seconds.")
                        raise RuntimeError(f"Server {server_name} did not become active within {timeout} seconds.")

                    ip = s.private_net[0].ip if s.private_net else s.public_net.ipv4.ip

                    logger.info(
                        f"Spawned Hetzner server {server_name} ({ip}) "
                        f"using {stype_name} (priority {stype_priority}) "
                        f"in {loc_name} (priority {loc_priority})."
                    )

                    # TODO health check

                    return {
                        "name": server_name,
                        "ip": ip,
                        "ssh_user": "root",
                        "ssh_key_path": self.ssh_private_key_path,
                        "server_id": server_id,
                        "server_type": stype.name,
                        "stype_priority": stype_priority,
                        "loc_priority": loc_priority,
                        "location": loc_name
                    }
                except RuntimeError as e:
                    msg = str(e)
                    if "did not become active" in msg:
                        # Re-raise the exception to propagate it
                        raise
                    else:
                        logger.warning(f"Failed {stype} in {loc_name}: {e}")
                        continue
                except Exception as e:
                    logger.warning(f"Failed {stype} in {loc_name}: {e}")
                    continue

        logger.error("All server types failed. Could not spawn backend.")
        raise RuntimeError("All server types failed. Could not spawn backend.")


    async def run_on_backend(self, backend: Dict[str, Any], prompt: str, timeout: int = 600) -> str:
        """
        Run inference via SSH tunnel to the HTTP server on the backend.

        Args:
            backend (dict): Backend information from spawn_backend.
            prompt (str): Prompt/question to process.
            timeout (int): Request timeout in seconds.

        Returns:
            str: Inference output.
        """
        logger.info(f"Running inference on backend '{backend['name']}' with ip '{backend['ip']}' for prompt '{prompt}'")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            HetznerProvisioner._with_ssh_tunnel,
            backend["ip"],
            backend["ssh_user"],
            backend["ssh_key_path"],
            prompt,
            timeout
        )

    @staticmethod
    def _with_ssh_tunnel(ip: str, user: str, key_path: str, prompt: str, timeout: int) -> str:
        """
        Open SSH tunnel to Hetzner backend and send an inference request.

        Args:
            ip (str): Public IP of the backend.
            user (str): SSH username.
            key_path (str): Path to SSH private key.
            prompt (str): Prompt/question to process.
            timeout (int): Timeout for HTTP request in seconds.

        Returns:
            str: Inference result.
        """
        logger.debug(f"Opening SSH tunnel to backend {ip}…")
        try:
            with SSHTunnel(
                (ip, 22),
                ssh_username=user,
                ssh_pkey=key_path,
                remote_bind_address=('127.0.0.1', 8000),
                local_bind_address=('127.0.0.1', 0)
            ) as tunnel:
                local_port = tunnel.local_bind_port
                logger.debug(f"Sending inference request to backend {ip} via local port {local_port}")
                resp = requests.post(
                    f"http://127.0.0.1:{local_port}/infer",
                    json={"prompt": prompt},
                    timeout=timeout
                )
                resp.raise_for_status()
                logger.debug(f"Received response from backend {ip}")
                return resp.text
        except Exception as e:
            logger.error(f"Error communicating with backend {ip}: {e}")
            return f"ERROR: {str(e)}"

    async def terminate_backend(self, backend: Dict[str, Any]) -> None:
        """
        Terminate the Hetzner backend server.

        Args:
            backend (dict): Backend information from spawn_backend.
        """
        server_id = backend.get("server_id")
        if not server_id:
            logger.error(f"No server ID found in backend info: {backend}")
            return

        server = self.client.servers.get_by_id(server_id)
        if server:
            private_ip = server.private_net[0].ip if server.private_net else "unknown"
            logger.info(f"Deleting Hetzner server {server.name} (private IP {private_ip})")
            server.delete()
        else:
            logger.error(f"Server {server_id} not found. It may have already been deleted.")
