"""
base_provisioner.py

Abstract base class for backend provisioners. Defines the interface that all
backend provisioners (Docker, Hetzner, etc.) must implement.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any

class BaseProvisioner(ABC):
    """
    Abstract base class for backend provisioners.

    A provisioner handles the lifecycle of backend instances, including
    starting, stopping, and executing inference. This interface ensures
    that the BackendPool can manage any provisioner in a uniform way.
    """

    @abstractmethod
    async def spawn_backend(self) -> Any:
        """
        Spawn a new backend instance (Docker container, Hetzner VM, etc.).

        Returns:
            Any: A handle or object representing the spawned backend instance.
        """
        pass

    @abstractmethod
    async def run_on_backend(self, backend: Any, prompt: str, timeout: int) -> str:
        """
        Execute inference on the given backend instance.

        Args:
            backend (Any): The backend instance returned by spawn_backend().
            prompt (str): The input question or prompt to process.
            timeout (int): The number of seconds to wait for a response.

        Returns:
            str: The generated output from the backend.
        """
        pass

    @abstractmethod
    async def terminate_backend(self, backend: Any) -> None:
        """
        Safely terminate the given backend instance.

        Args:
            backend (Any): The backend instance to terminate.

        Returns:
            None
        """
        pass

    @abstractmethod
    async def healthcheck(self, id: Any) -> None:
        """
        Checks the health of the instance, i.e., whether the inference server is up and running.
        Raises an TimeoutError if the inference server is not running.
        :param id: Usually a port (docker) or IP address (hetzner).
        :return: None
        """
        pass


# --- Protocol Definition ---

class Provisioner(Protocol):
    """
    Protocol for a backend provisioner.
    All provisioners (Docker, Hetzner, etc.) must implement these async methods.
    """

    async def spawn_backend(self) -> Any:
        """
        Spawn a new backend instance.
        Returns a backend object that can be passed to `run_on_backend`.
        """
        ...

    async def run_on_backend(self, backend: Any, prompt: str, timeout: int) -> str:
        """
        Run inference on a given backend instance.
        Args:
            backend: Backend instance returned from `spawn_backend`.
            prompt: The prompt/question to process.
            timeout: Timeout to wait for inference.
        Returns:
            str: Inference output.
        """
        ...

    async def terminate_backend(self, backend: Any) -> None:
        """
        Terminate a backend instance.
        Args:
            backend: Backend instance to terminate.
        """
        ...

    async def healthcheck(self, id: Any) -> None:
        """
        Checks the health of the instance, i.e., whether the inference server is up and running.
        Raises an TimeoutError if the inference server is not running.
        :param id: Usually a port (docker) or IP address (hetzner).
        :return: None
        """
        ...
