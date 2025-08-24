"""
backend_pool_singleton.py

Thread-safe, asynchronous singleton backend pool for managing inference instances.

Supports:
- Maximum concurrent backends
- Idle termination
- Queuing requests when all backends are busy
- Automatic release via context manager
- Any provisioner implementing BaseProvisioner (Docker, Hetzner, etc.)
"""
import os
import asyncio
from typing import Any, Dict, List, Optional, TypeVar, AsyncGenerator

import math

from app.llm_worker_provisioners import BaseProvisioner, Provisioner
from app import logger
from contextlib import asynccontextmanager

# Type of object returned by a provisioner
Backend = TypeVar("Backend")


class BackendInstance:
    """
    Active Backend Handle. For an easier interface and interaction. This is how instances are provided to user.
    Unified wrapper around a backend instance and its provisioner.

    Provides a simple `.run_on_backend(prompt)` interface for any type of backend.
    """

    def __init__(self, provisioner: Provisioner, backend: Backend, inference_timeout: int, size:str="large"):
        self.provisioner = provisioner
        self.backend = backend
        self._inference_timeout = inference_timeout
        self.size = size

    @property
    def is_small(self) -> bool:
        return self.size == "small"

    async def run_on_backend(self, prompt: str) -> str:
        """Run a prompt on this backend instance."""
        return await self.provisioner.run_on_backend(self.backend, prompt, timeout=self._inference_timeout)

    async def terminate(self) -> None:
        """Terminate this backend instance."""
        try:
            await self.provisioner.terminate_backend(self.backend)
        except Exception as e:
            logger.error(f"Failed to terminate backend: {e}")


class BackendInstanceWrapper:
    """
    Pool-related metadata
    Wraps a backend instance returned by a provisioner for bookkeeping.

    Tracks whether the backend is busy and last used timestamp.
    """

    def __init__(self, instance: Backend, size="large"):
        now = asyncio.get_running_loop().time()
        self.instance: Backend = instance
        self.busy: bool = False
        # self.is_cold: bool = True  # TODO this would be required for even better time and resource management
        self.last_used: float = now
        self.start_time: float = now
        if size not in ["large", "small"]:
            raise ValueError(f"Invalid backend size: {size}")
        self.size: str = size

    @property
    def is_small(self) -> bool:
        return self.size == "small"

class BackendPool:
    """
    Pool managing backends for inference.

    Automatically spawns, queues, and cleans up idle backends.

    Attributes:
        provisioner: Instance of a BaseProvisioner to handle backend creation/termination.
        max_size: Maximum number of concurrent backends.
        idle_buffer: Safety buffer in seconds before the next paid hour to terminate idle backends.
        inference_timeout: Timeout for a single inference request.
        backends: List of currently managed BackendWrapper instances.
        lock: Asyncio lock for thread-safe operations.
        queue: Queue for pending requests if all backends are busy.
        _recent_request_times: Tuple of lists ([startup_times], [inference_times]) to estimate averages.
    """

    def __init__(self,
                 provisioner: BaseProvisioner,
                 max_size: int = 5,
                 idle_buffer: int = 360,
                 inference_timeout: int = 300):
        self.provisioner: BaseProvisioner = provisioner
        self.max_size: int = max_size
        self.idle_buffer: int = idle_buffer
        self.inference_timeout: int = inference_timeout
        self.backends: List[BackendInstanceWrapper] = []
        self.lock: asyncio.Lock = asyncio.Lock()
        self.queue: asyncio.Queue[asyncio.Future] = asyncio.Queue()
        self._idle_task: Optional[asyncio.Task] = None
        self.billing_interval = 30 * 60

        # Tuple of lists: ([startup_times], [inference_times])
        self._recent_request_times: tuple[list[float], list[float]] = ([], [])

        logger.debug(f"Set inference_timeout to {self.inference_timeout}")

    async def shutdown_all_backends(self):
        """Terminate all backends sequentially and log each termination."""
        logger.info("Shutting down all backends…")
        for bw in list(self.backends):  # use list() to avoid mutation issues
            try:
                await self.provisioner.terminate_backend(bw.instance)
                logger.info(f"Backend {bw.instance} terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating backend {bw.instance}: {e}")
            finally:
                self.backends.remove(bw)
        logger.info("All backends terminated")

    def __del__(self):
        """Ensure shutdown on object deletion."""
        # Only run shutdown if event loop is running
        try:
            loop = asyncio.get_running_loop()
            # schedule shutdown in the running loop
            loop.create_task(self.shutdown_all_backends())
        except RuntimeError:
            # No running loop, safe to create a new one
            asyncio.run(self.shutdown_all_backends())

    def get_capacity(self) -> Dict[str, int]:
        capacity = {
            "max_size": self.max_size,
            "current_size": len(self.backends),
            "queued_requests": self.queue.qsize(),
        }
        capacity["current_capacity"] = capacity["max_size"] - capacity["current_size"]
        return capacity


    def has_ready_instance(self) -> bool:
        """
        Return True if there is at least one idle backend ready to be used.
        """
        # No lock needed here if you accept a slightly stale view
        for bw in self.backends:
            if not bw.busy:
                return True
        return False

    async def init(self, warm_backend:Backend = None) -> None:
        """Initialize the pool, starting the idle cleanup loop."""
        async with self.lock:
            if not self._idle_task:
                self._idle_task = asyncio.create_task(self._idle_cleanup_loop())
            if warm_backend:
                if self.backends:
                    raise AssertionError("Tried to initialize but seems already initialized. "
                                         "There are already backends.")
                logger.debug(f"Checking backend {warm_backend}")
                await self.provisioner.health_check(
                    warm_backend.get(
                        "ip",
                        warm_backend.get("port", None)
                    )
                )
                logger.debug(f"Health check succeeded for {warm_backend}")
                self.backends.append(BackendInstanceWrapper(warm_backend, size="small"))


    async def get_average_times(self) -> tuple[float, float]:
        """
        Compute and return the average startup and processing (inference) times.

        Uses recent recorded times if available; otherwise, falls back to default values:
            - Default startup time: 7 minutes
            - Default inference time: 3 minutes

        Returns:
            tuple[float, float]: (average_startup_seconds, average_inference_seconds)

        Note:
            This method does NOT acquire self.lock internally.
            The caller should hold the lock if consistent reading with other state is required.
        """
        startup_times, inference_times = self._recent_request_times
        avg_startup = sum(startup_times) / len(startup_times) if startup_times else 10 * 60
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 3 * 60
        return avg_startup, avg_inference


    async def should_spawn_new_backend(self) -> bool:
        """
        Decide whether to spawn a new backend based on queue length and times.

        Returns True if starting a new backend reduces estimated wait.

        Note:
            This method does NOT acquire self.lock internally.
            The caller should hold the lock if consistent reading with other state is required.
        """
        if not self.backends:
            return True  # Always spawn at least one backend

        queued_count = self.queue.qsize()
        busy_backends = len([bw for bw in self.backends if bw.busy])
        if (len(self.backends) - busy_backends) > 0:
            return False  # There is at least one idling backend. Give it the task!
        # Starting from here: There are backends, and all are busy.

        avg_startup, avg_inference = await self.get_average_times()

        # Total wait time if we do NOT spawn: first in queue waits for N backends * avg_inference
        estimated_wait_no_spawn = avg_inference * (math.ceil(queued_count / busy_backends) + 1)

        # Total wait time if we spawn one new backend
        estimated_wait_spawn = avg_startup + avg_inference  # first inference will run after startup

        # Economic decision: spawn if waiting is longer than startup cost
        return estimated_wait_no_spawn > estimated_wait_spawn


    async def estimate_wait_time(self) -> float:
        """
        Estimate wait time in seconds until a backend is available.

        Logic:
          - If at least one idle backend exists, return avg_inference.
          - If no idle backends, check should_spawn_new_backend:
              - If spawn justified: avg_startup + avg_inference
              - Else: (queued_count + 1) / total_instances * avg_inference

        Returns:
            float: estimated wait time in seconds
        """
        async with self.lock:
            idle_backends = [bw for bw in self.backends if not bw.busy]
            queued_count = self.queue.qsize()
            total_instances = max(1, len(self.backends))  # avoid div by zero
            avg_startup, avg_inference = await self.get_average_times()

            if idle_backends:
                return avg_inference
            else:
                if await self.should_spawn_new_backend():
                    return avg_startup + avg_inference
                else:
                    return (queued_count + 1) / total_instances * avg_inference

    async def record_request_time(self, startup_sec: float = 0, inference_sec: float = 0, use_lock:bool = True) -> None:
        """
        Record the duration of a backend request for tracking average times.

        Maintains two rolling lists (startup times and inference times) with a maximum
        of 100 entries each, used to estimate backend performance and economic decisions.

        Args:
            startup_sec (float, optional): Time in seconds taken to start a backend instance.
                Only recorded if greater than 0. Defaults to 0.
            inference_sec (float, optional): Time in seconds taken to process a single inference request.
                Only recorded if greater than 0. Defaults to 0.
            use_lock (bool, optional): If True (default), acquires the internal async lock to
                ensure thread-safe updates. If False, caller must guarantee thread safety.

        Notes:
            - The internal lock should be used unless the caller already holds it.
            - Both startup_sec and inference_sec are optional, but at least one should be > 0
              to affect recorded averages.
            - This method updates internal lists and trims them to the 100 most recent entries
              to avoid unbounded memory growth and keep a moving average.
        """
        def _action(startup_sec: float, inference_sec: float) -> None:
            startup_times, inference_times = self._recent_request_times

            if startup_sec > 0:
                startup_times.append(startup_sec)
            if inference_sec > 0:
                inference_times.append(inference_sec)

            # Keep history limited
            if len(startup_times) > 100:
                startup_times = startup_times[-100:]
            if len(inference_times) > 100:
                inference_times = inference_times[-100:]

            self._recent_request_times = (startup_times, inference_times)

        if use_lock:
            async with self.lock:
                _action(startup_sec, inference_sec)
        else:
            _action(startup_sec, inference_sec)

    async def get_backend(self) -> BackendInstanceWrapper:
        """
        Acquire an available backend or wait in the queue if all are busy.
        """
        async with self.lock:
            if not self._idle_task:
                await self.init()

            # Prefer large over small if both available
            idle_large = next((bw for bw in self.backends if not bw.busy and not bw.is_small),
                              None)
            if idle_large:
                idle_large.busy = True
                idle_large.last_used = asyncio.get_running_loop().time()
                return idle_large

            idle_small = next((bw for bw in self.backends if not bw.busy and bw.is_small),
                              None)
            if idle_small:
                idle_small.busy = True
                idle_small.last_used = asyncio.get_running_loop().time()
                return idle_small

            # Spawn new backend if under max size
            if len(self.backends) < self.max_size:
                if await self.should_spawn_new_backend():
                    logger.info("Economically justified: Spawning new backend")
                    start = asyncio.get_event_loop().time()
                    instance = await self.provisioner.spawn_backend()
                    startup_duration = asyncio.get_event_loop().time() - start
                    logger.info(f"New backend ready. Time passed: {startup_duration:.1f} seconds")
                    bw = BackendInstanceWrapper(instance)
                    bw.busy = True
                    self.backends.append(bw)
                    await self.record_request_time(startup_sec=startup_duration, use_lock=False)
                    return bw
                else:
                    logger.info("Spawn of backend not economically justified. Putting into queue.")
            else:
                logger.info(f"Maximum pool size of {self.max_size} reached. "
                            f"Putting into queue with length of {self.queue.qsize()}")

        # Queue request if all backends busy
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self.queue.put(future)
        return await future

    async def release_backend(self, bw: BackendInstanceWrapper) -> None:
        """
        Release a backend, mark it idle, and serve queued requests if any.
        """
        async with self.lock:
            bw.busy = False
            bw.last_used = asyncio.get_running_loop().time()

            while not self.queue.empty():
                if bw.busy:  # Already assigned
                    break
                future = await self.queue.get()
                if not future.done():
                    bw.busy = True
                    future.set_result(bw)

    async def _idle_cleanup_loop(self) -> None:
        """
        Terminate idle backends that are no longer economically justified.

        This loop runs periodically (every 60 seconds) and checks each backend
        to see if it should be terminated based on:
            1. Whether it is idle (not busy)
            2. How long it has been idle
            3. How close it is to the next paid hour, minus a buffer
            4. Economic justification: average startup + inference time

        The goal is to avoid paying for unnecessary compute time
        while keeping enough instances available for low wait times.
        """
        while True:
            resolution = 60
            await asyncio.sleep(resolution)  # Wait 1 minute between checks
            now = asyncio.get_event_loop().time()  # Current loop time in seconds

            async with self.lock:
                # Compute economic idle timeout based on recent averages
                # This represents the total time we "waste" by keeping an idle instance
                avg_startup, avg_inference = await self.get_average_times()
                economic_idle_timeout = avg_startup + avg_inference

                to_remove: List[BackendInstanceWrapper] = []

                for bw in self.backends:
                    if bw.busy or bw.is_small:
                        # Skip backends that are currently processing a request
                        # And never delete the small, hot instances
                        continue

                    idle_duration = now - bw.last_used
                    # Time since backend was originally started
                    time_since_start = now - bw.start_time

                    # Calculate seconds until the next paid hour
                    # e.g., if instance started at 12:20, next paid hour ends at 13:20
                    next_billing = self.billing_interval - (time_since_start % self.billing_interval)

                    # Decide to remove the backend only if BOTH conditions are true:
                    # 1. Idle duration is longer than the economic threshold (avg_startup + avg_inference)
                    # 2. We are within the "idle buffer" window before the next paid hour
                    logger.debug(f"idle duration: {idle_duration:.1f} seconds, "
                                 f"avg startup: {avg_startup:.1f} seconds, "
                                 f"avg inference: {avg_inference:.1f} seconds, "
                                 f"economic_idle_timeout: {economic_idle_timeout:.1f}, "
                                 f"next billing in: {next_billing:.1f} seconds. "
                                 f"Decision: {idle_duration:.1f} > {economic_idle_timeout:.1f} AND "
                                 f"{next_billing:.1f} <= {self.idle_buffer + resolution}: "
                                 f"{idle_duration > economic_idle_timeout and next_billing <= self.idle_buffer + resolution}.")
                    if idle_duration > economic_idle_timeout and next_billing <= self.idle_buffer + resolution:
                        to_remove.append(bw)

                # Terminate all backends that matched the criteria
                for bw in to_remove:
                    idle_duration = now - bw.last_used
                    time_since_start = now - bw.start_time
                    next_billing = self.billing_interval - (time_since_start % self.billing_interval)
                    logger.info(
                        f"Terminating idle backend (idle {idle_duration:.0f}s (={idle_duration/60:.1f}m), {next_billing:.0f}s until next paid hour)")
                    await self.provisioner.terminate_backend(bw.instance)
                    self.backends.remove(bw)

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[BackendInstance, Any]:
        """
        Async context manager to automatically acquire and release a backend.

        Usage:
            async with pool.acquire() as backend:
                result = await backend.run_on_backend("prompt")
        """
        bw = await self.get_backend()
        backend_instance = BackendInstance(self.provisioner,
                                           bw.instance,
                                           inference_timeout=self.inference_timeout,
                                           size=bw.size)
        try:
            yield backend_instance
        finally:
            await self.release_backend(bw)


class SingletonBackendPool:
    """
    Thread-safe singleton accessor for BackendPool.
    """
    _instance: Optional[BackendPool] = None
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, config: Dict[str, Any]) -> BackendPool:
        """
        Return singleton BackendPool instance, creating it if necessary.

        Args:
            config: Dictionary with keys:
                - 'mode': 'local' or 'hetzner'
                - 'backend': dict with max_backends, idle_timeout
                - provider-specific keys
        """
        logger.debug(f"Config for Backend Pool: {config}")
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    mode = config.get("mode", "local")
                    backend_cfg = config.get("backend", {})
                    max_size = backend_cfg.get("max_backends", 5)
                    idle_buffer = backend_cfg.get("idle_buffer", 360)

                    if mode == "hetzner":
                        from app.llm_worker_provisioners.hetzner import HetznerProvisioner
                        hetzner_cfg = config.get("hetzner", {})
                        api_token = os.environ.get("HETZNER_API_TOKEN")
                        if not api_token:
                            raise RuntimeError("HETZNER_API_TOKEN not set")

                        provisioner = HetznerProvisioner(
                            api_token=api_token,
                            snapshot_name=hetzner_cfg['snapshot_name'],
                            ssh_key_name=hetzner_cfg['ssh_key_name'],
                            server_types=hetzner_cfg.get('server_types', ["CX52", "CX42", "CPX41", "CX32", "CPX31"]),
                            ssh_private_key_path=os.path.expanduser(hetzner_cfg['ssh_key_path']),
                            private_network_name=hetzner_cfg['private_network_name'],
                        )
                    else:
                        from app.llm_worker_provisioners.local_docker import LocalDockerProvisioner
                        provisioner = LocalDockerProvisioner(
                            health_timeout=backend_cfg.get("health_timeout", 600.0)
                        )

                    cls._instance = BackendPool(
                        provisioner=provisioner,
                        max_size=max_size,
                        idle_buffer=idle_buffer,
                        inference_timeout=backend_cfg.get("inference_timeout", 300)
                    )
                    warm_backend = backend_cfg.get("warm_backend", None)
                    await cls._instance.init(warm_backend=warm_backend)
        return cls._instance
