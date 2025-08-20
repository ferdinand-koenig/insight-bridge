"""
local_docker.py

Async provisioner for running inference in local Docker containers.
Implements BaseProvisioner interface.
"""

import asyncio

import docker
import httpx
from pathlib import Path
from .base_provisioner import BaseProvisioner
from app import logger

class LocalDockerProvisioner(BaseProvisioner):
    """
    Provisioner for local Docker-based backends.

    Spawns Docker containers with preloaded models, runs inference, and
    terminates containers when no longer needed.
    """

    def __init__(self, image: str = "insight-bridge-worker:latest", health_timeout: float = 120.0):
        """
        Args:
            image (str): Docker image name to use for inference.
            health_timeout (float): Timeout for waiting for the health check (when Container is up and running)
        """
        self.image = image
        self.client = docker.from_env()
        self.health_timeout = health_timeout

    async def spawn_backend(self) -> dict:
        """
        Spawn a new Docker container for inference and wait until it's healthy.

        Returns:
            dict: Contains container instance and host port.
        """
        # Run container detached with dynamic host port binding
        container = self.client.containers.run(
            self.image,
            detach=True,
            tty=True,
            auto_remove=False,
            ports={"8000/tcp": None},  # Docker assigns a free host port
            volumes={
                str(Path("./model").resolve()) : {"bind": "/model", "mode": "rw"},
                str(Path("./config.yaml").resolve()): {"bind": "/insight-bridge/config.yaml", "mode": "rw"},
                str(Path("./prompt_template.yaml").resolve()): {"bind": "/insight-bridge/prompt_template.yaml", "mode": "rw"},
            }
        )

        # Retrieve the dynamically assigned host port
        container.reload()  # refresh container.attrs
        host_port = int(container.attrs['NetworkSettings']['Ports']['8000/tcp'][0]['HostPort'])

        # Wait until the container is ready by polling /health
        url = f"http://127.0.0.1:{host_port}/health"
        timeout = self.health_timeout  # seconds
        interval = 1
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(url, timeout=1.0)
                    if resp.status_code == 200:
                        logger.info(f"Docker at '{url}' got ready. Webservice's response: 200 - ok.")
                        break
                except Exception:
                    pass
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise TimeoutError(f"Worker at port {host_port} did not become ready in {timeout}s")
                await asyncio.sleep(interval)

        return {"container": container, "host_port": host_port}

    async def run_on_backend(self, backend: dict, prompt: str, timeout: int=300) -> str:
        """
        Run inference on a backend container via HTTP request.

        Args:
            backend (dict): Container info with 'container' and 'host_port'.
            prompt (str): Prompt/question to process.
            timeout (int): timeout for inference

        Returns:
            str: Inference output.
        """
        url = f"http://127.0.0.1:{backend['host_port']}/infer"
        params = {"question": prompt}
        logger.info(f"Sending query to '{url}' with params: {params} with a timeout of {timeout}s")
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                return resp.text
            except httpx.ReadTimeout:
                logger.error(f"Request {url}:'{params}' timed out after {timeout} seconds.")
                return "TIMEOUT of inference."  # or None, or a custom sentinel value
            except httpx.HTTPStatusError as e:
                logger.error(f"Backend returned error {e.response.status_code}: {e.response.text}")
                return f"ERROR: {e.response.status_code}"
            except Exception as e:
                logger.error(f"Unexpected error while querying backend: {e}")
                return f"ERROR: {str(e)}"

    async def terminate_backend(self, backend: dict) -> None:
        """
        Stop and remove the Docker container.

        Args:
            backend (dict): Container info with 'container' instance.
        """
        container = backend["container"]
        try:
            container.stop(timeout=5)
            container.remove()
        except Exception as e:
            logger.warn(f"Failed to terminate Docker container: {e}")
