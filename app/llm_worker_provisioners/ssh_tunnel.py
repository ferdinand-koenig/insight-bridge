# ssh_tunnel.py

import paramiko  # TODO import explicitely
import socket
from app import logger

class SSHTunnel:
    def __init__(self, server_address, ssh_username, ssh_pkey,
                 remote_bind_address=('127.0.0.1', 8000),
                 local_bind_address=('127.0.0.1', 0)):
        """
        Mimic SSHTunnelForwarder interface using paramiko directly.

        Args:
            server_address (tuple): (host, port)
            ssh_username (str): SSH username
            ssh_pkey (str or paramiko.PKey): Path to private key file or PKey object
            remote_bind_address (tuple): Remote address to forward to (host, port)
            local_bind_address (tuple): Local address to bind (host, port), port=0 picks random
        """
        self.host, self.ssh_port = server_address
        self.username = ssh_username
        self.key = ssh_pkey
        self.remote_bind_address = remote_bind_address
        self.local_bind_address = local_bind_address

        self.client = None
        self.transport = None
        self.local_bind_port = None
        self._sock = None
        self._channel = None

    def start(self):
        """Start the SSH tunnel."""
        # Load key if given as string path
        if isinstance(self.key, str):
            key_obj = None
            for key_cls in [paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey]:
                try:
                    key_obj = key_cls.from_private_key_file(self.key)
                    logger.debug(f"Successfully loaded {key_cls.__name__} from {self.key}")
                    break
                except paramiko.ssh_exception.SSHException:
                    continue
            if not key_obj:
                logger.error(f"Could not load private key {self.key}")
                raise RuntimeError(f"Could not load private key {self.key}")
            self.key = key_obj

        # Connect SSH
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.host,
            port=self.ssh_port,
            username=self.username,
            pkey=self.key
        )
        self.transport = self.client.get_transport()

        # Bind a local port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(self.local_bind_address)
        self.local_bind_port = self._sock.getsockname()[1]
        self._sock.listen(1)
        logger.debug(f"Local port {self.local_bind_port} bound for forwarding")

        # Open channel for remote forwarding
        self._channel = self.transport.open_channel(
            "direct-tcpip",
            dest_addr=self.remote_bind_address,
            src_addr=self._sock.getsockname()
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._channel:
            self._channel.close()
        if self._sock:
            self._sock.close()
        if self.client:
            self.client.close()
        logger.debug("SSHTunnelForwarder closed")
