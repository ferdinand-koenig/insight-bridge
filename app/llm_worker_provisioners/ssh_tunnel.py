# ssh_tunnel.py

import paramiko
import socket
import threading
import select
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
        self._stop_event = None
        self._thread = None

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
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.local_bind_address)
        self.local_bind_port = self._sock.getsockname()[1]
        self._sock.listen(5)
        logger.debug(f"Local port {self.local_bind_port} bound for forwarding")

        # Start forwarding thread
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._forward_loop, daemon=True)
        self._thread.start()

    def _forward_loop(self):
        while not self._stop_event.is_set():
            try:
                r_sock, _ = self._sock.accept()
                threading.Thread(target=self._handle_client, args=(r_sock,), daemon=True).start()
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error accepting connection: {e}")

    def _handle_client(self, client_sock):
        try:
            chan = self.transport.open_channel(
                "direct-tcpip",
                dest_addr=self.remote_bind_address,
                src_addr=client_sock.getsockname()
            )
        except Exception as e:
            logger.error(f"Could not open channel: {e}")
            client_sock.close()
            return

        # Forward data both ways
        try:
            while True:
                rlist, _, _ = select.select([client_sock, chan], [], [])
                if client_sock in rlist:
                    data = client_sock.recv(1024)
                    if len(data) == 0:
                        break
                    chan.sendall(data)
                if chan in rlist:
                    data = chan.recv(1024)
                    if len(data) == 0:
                        break
                    client_sock.sendall(data)
        finally:
            chan.close()
            client_sock.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stop_event:
            self._stop_event.set()
        if self._sock:
            self._sock.close()
        if self.client:
            self.client.close()
        logger.debug("SSHTunnel closed")
