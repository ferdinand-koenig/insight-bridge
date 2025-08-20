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
from sshtunnel import SSHTunnelForwarder
from hcloud import Client

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

    async def spawn_backend(self) -> Dict[str, Any]:
        """
        Spawn a Hetzner Cloud server using the specified snapshot and SSH key.

        Returns:
            dict: Backend information including IP, SSH user, SSH key path, and server ID.
        """
        # Find snapshot
        snapshots = self.client.images.get_all(type="snapshot")
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

        for priority, stype in enumerate(self.server_types, start=1):
            try:
                server_name = f"llm-instance-{int(asyncio.get_event_loop().time())}"
                server = self.client.servers.create(
                    name=server_name,
                    server_type=stype,
                    image=snapshot.id,
                    ssh_keys=[key.id],
                    networks=[net.id],  # ensure attached to private network
                    wait_for_active=True,
                    public_net=False,
                )
                logger.info(
                    f"Spawned Hetzner server {server_name} "
                    f"({server.private_net[0].ip if server.private_net else server.public_net.ipv4.ip}) "
                    f"using {stype} (priority {priority})"
                )
                return {
                    "ip": server.private_net[0].ip,
                    "ssh_user": "root",
                    "ssh_key_path": self.ssh_private_key_path,
                    "server_id": server.id,
                    "server_type": stype,
                    "priority": priority
                }
            except Exception as e:
                logger.warning(f"Failed to spawn server type {stype} (priority {priority}): {e}")
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
            with SSHTunnelForwarder(
                (ip, 22),
                ssh_username=user,
                ssh_private_key=key_path,
                remote_bind_address=('127.0.0.1', 8000),
                local_bind_address=('127.0.0.1', 0)
            ) as tunnel:
                tunnel.start()
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
