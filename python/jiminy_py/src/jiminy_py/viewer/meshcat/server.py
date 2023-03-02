""" TODO: Write documentation.
"""
import os
import sys
import signal
import logging
import asyncio
import argparse
import multiprocessing
import multiprocessing.managers
from typing import Optional, Tuple, Sequence, Dict, List, Set

import zmq
import psutil
import umsgpack
import tornado.web
import tornado.ioloop
from zmq.eventloop.zmqstream import ZMQStream

from meshcat.servers.tree import walk, find_node
from meshcat.servers.zmqserver import (
    DEFAULT_ZMQ_METHOD, VIEWER_ROOT, StaticFileHandlerNoCache,
    ZMQWebSocketBridge, WebSocketHandler, find_available_port)


DEFAULT_COMM_PORT = 6500
WAIT_COM_TIMEOUT = 5.0  # in seconds


# Disable tornado access error logging
logging.getLogger('tornado.access').disabled = True

# ================ Monkey-patch =======================

# Add support of cross-origin connection.
# It is useful to execute custom javascript commands within a Jupyter Notebook,
# and it is not an actual security flaw for local servers since they are not
# accessible from the outside anyway.
WebSocketHandler.check_origin = lambda self, origin: True


# Override the default html page to disable auto-update of three js "controls"
# of the camera, so that it can be moved programmatically in any position,
# without any constraint, as long as the user is not moving it manually using
# the mouse.
class MyFileHandler(StaticFileHandlerNoCache):
    """ TODO: Write documentation.
    """
    def initialize(self,  # pylint: disable=arguments-differ
                   default_path: str,
                   default_filename: str,
                   fallback_path: str) -> None:
        """ TODO: Write documentation.
        """
        # pylint: disable=attribute-defined-outside-init
        self.default_path = os.path.abspath(default_path)
        self.default_filename = default_filename
        self.fallback_path = os.path.abspath(fallback_path)
        super().initialize(self.default_path, self.default_filename)

    def validate_absolute_path(self,
                               root: str,
                               absolute_path: str) -> Optional[str]:
        """ TODO: Write documentation.
        """
        if os.path.isdir(absolute_path):
            if not self.request.path.endswith("/"):
                self.redirect(self.request.path + "/", permanent=True)
                return None
            absolute_path = os.path.join(absolute_path, self.default_filename)
            return self.validate_absolute_path(root, absolute_path)
        if os.path.exists(absolute_path) and \
                os.path.basename(absolute_path) != self.default_filename:
            return super().validate_absolute_path(root, absolute_path)
        return os.path.join(
            self.fallback_path, absolute_path[(len(root) + 1):])


# Implement bidirectional communication because zmq and the websockets by
# gathering and forward messages received from the websockets to zmq. Note that
# there is currently no way to identify the client associated to each reply,
# but it is usually not a big deal, since the same answers is usually
# expected from each of them. Comma is used as a delimiter.
#
# It also fixes flushing issue when 'handle_zmq' is not directly responsible
# for sending a message through the zmq socket.
def handle_web(self: WebSocketHandler, message: str) -> None:
    """ TODO: Write documentation.
    """
    # Ignore watchdog for websockets since it would be closed if non-responding
    if message != 'meshcat:watchdog':
        self.bridge.websocket_msg.append(message)
    # It should not be necessary to check 'is_waiting_ready_msg' because this
    # method should never be triggered if there is nothing to send, but still
    # doing it in case of some unexpected edge-case.
    if not self.bridge.is_waiting_ready_msg:
        return
    # Send acknowledgement if everybody replied
    if len(self.bridge.websocket_msg) == len(self.bridge.websocket_pool) and \
            len(self.bridge.comm_msg) == len(self.bridge.comm_pool):
        self.bridge.is_waiting_ready_msg = False
        gathered_msg = ",".join(
            self.bridge.websocket_msg + list(self.bridge.comm_msg.values()))
        self.bridge.zmq_socket.send(gathered_msg.encode("utf-8"))
        self.bridge.zmq_stream.flush()
        self.bridge.comm_msg = {}
        self.bridge.websocket_msg = []
WebSocketHandler.on_message = handle_web  # noqa


class ZMQWebSocketIpythonBridge(ZMQWebSocketBridge):
    """ TODO: Write documentation.
    """
    def __init__(self,
                 zmq_url: Optional[str] = None,
                 comm_url: Optional[str] = None,
                 host: str = "127.0.0.1",
                 port: Optional[int] = None):
        super().__init__(zmq_url, host, port)

        # Create a new zmq socket specifically for kernel communications
        if comm_url is None:
            def f(port: int) -> Tuple[zmq.Socket, ZMQStream, str]:
                return self.setup_comm(
                    f"{DEFAULT_ZMQ_METHOD}://{self.host}:{port}")
            (self.comm_zmq, self.comm_stream, self.comm_url), _ = \
                find_available_port(f, DEFAULT_COMM_PORT)
        else:
            self.comm_zmq, self.comm_stream, self.comm_url = \
                 self.setup_comm(comm_url)

        # Extra buffers for: comm ids and messages
        self.comm_pool: Set[bytes] = set()
        self.watch_pool: Set[bytes] = set()
        self.comm_msg: Dict[bytes, str] = {}
        self.websocket_msg: List[str] = []
        self.is_waiting_ready_msg = False

        # Start the comm watchdog
        self.watchdog_comm()

    def setup_comm(self, url: str) -> Tuple[zmq.Socket, ZMQStream, str]:
        """ TODO: Write documentation.
        """
        comm_zmq = self.context.socket(zmq.XREQ)
        comm_zmq.bind(url)
        comm_stream = ZMQStream(comm_zmq)
        comm_stream.on_recv(self.handle_comm)
        return comm_zmq, comm_stream, url

    def make_app(self) -> tornado.web.Application:
        """ TODO: Write documentation.
        """
        return tornado.web.Application([
            (r"/static/?(.*)", MyFileHandler, {
                "default_path": VIEWER_ROOT,
                "fallback_path": os.path.dirname(__file__),
                "default_filename": "index.html"}),
            (r"/", WebSocketHandler, {"bridge": self})
        ])

    def wait_for_websockets(self) -> None:
        """ TODO: Write documentation.
        """
        if self.websocket_pool or self.comm_pool:
            self.zmq_socket.send(b"ok")
            self.zmq_stream.flush()
        else:
            self.ioloop.call_later(0.1, self.wait_for_websockets)

    def watchdog_comm(self) -> None:
        """ TODO: Write documentation.
        """
        # Purge non-responding comms
        for comm_id in self.comm_pool.copy():
            if comm_id not in self.watch_pool:
                self.comm_pool.discard(comm_id)
                self.comm_msg.pop(comm_id, None)
        self.watch_pool.clear()

        # Trigger ready sending if comm has been purged
        if self.is_waiting_ready_msg and \
                len(self.websocket_msg) == len(self.websocket_pool) and \
                len(self.comm_msg) == len(self.comm_pool):
            self.is_waiting_ready_msg = False
            gathered_msg = ",".join(
                self.websocket_msg + list(self.comm_msg.values()))
            self.zmq_socket.send(gathered_msg.encode("utf-8"))
            self.zmq_stream.flush()
            self.comm_msg = {}
            self.websocket_msg = []

        self.ioloop.call_later(WAIT_COM_TIMEOUT, self.watchdog_comm)

    def handle_zmq(self, frames: Sequence[bytes]) -> None:
        """ TODO: Write documentation.
        """
        cmd = frames[0].decode("utf-8")
        if cmd == "ready":
            self.comm_stream.flush()
            if not self.websocket_pool and not self.comm_pool:
                self.zmq_socket.send(b"")
                self.zmq_stream.flush()
                return
            msg = umsgpack.packb({"type": "ready"})
            self.is_waiting_ready_msg = True
            for websocket in self.websocket_pool:
                websocket.write_message(msg, binary=True)
            for comm_id in self.comm_pool:
                self.forward_to_comm(comm_id, msg)
        elif cmd == "list":
            # Only set_transform command is supported for now
            for i in range(1, len(frames), 3):
                _cmd, path, data = frames[i:(i+3)]
                path = list(filter(None, path.decode("utf-8").split("/")))
                find_node(self.tree, path).transform = data
                super().forward_to_websockets(frames[i:(i+3)])
            for comm_id in self.comm_pool:
                self.comm_zmq.send_multipart([comm_id, *frames[3::3]])
            self.zmq_socket.send(b"ok")
            self.zmq_stream.flush(zmq.POLLOUT)
        elif cmd == "stop":
            self.ioloop.stop()
        else:
            super().handle_zmq(frames)

    def handle_comm(self, frames: Sequence[bytes]) -> None:
        """ TODO: Write documentation.
        """
        cmd = frames[0].decode("utf-8")
        comm_id = cmd.split(':', 2)[1].encode()
        if cmd.startswith("open:"):
            self.send_scene(comm_id=comm_id)
            self.comm_pool.add(comm_id)
            self.watch_pool.add(comm_id)
            if self.is_waiting_ready_msg:
                # Send request for acknowledgment a-posteriori
                msg = umsgpack.packb({"type": "ready"})
                self.forward_to_comm(comm_id, msg)
        elif cmd.startswith("close:"):
            # Using `discard` over `remove` to avoid raising exception if
            # 'comm_id' is not found. It may happened if an old comm is closed
            # after Jupyter-notebook reset for instance.
            self.comm_pool.discard(comm_id)
            self.comm_msg.pop(comm_id, None)
        elif cmd.startswith("data:"):
            # Extract the message
            message = cmd.split(':', 2)[2]
            # Catch watchdog messages
            if message == "watchdog":
                self.watch_pool.add(comm_id)
                return
            if comm_id in self.comm_pool:
                # The comm may have already been thrown away already
                self.comm_msg[comm_id] = message
        if self.is_waiting_ready_msg and \
                len(self.websocket_msg) == len(self.websocket_pool) and \
                len(self.comm_msg) == len(self.comm_pool):
            self.is_waiting_ready_msg = False
            gathered_msg = ",".join(
                self.websocket_msg + list(self.comm_msg.values()))
            self.zmq_socket.send(gathered_msg.encode("utf-8"))
            self.zmq_stream.flush()
            self.comm_msg = {}
            self.websocket_msg = []

    def forward_to_websockets(self, frames: Sequence[bytes]) -> None:
        """ TODO: Write documentation.
        """
        super().forward_to_websockets(frames)
        *_, data = frames
        for comm_id in self.comm_pool:
            self.forward_to_comm(comm_id, data)

    def forward_to_comm(self, comm_id: bytes, message: bytes) -> None:
        """ TODO: Write documentation.
        """
        self.comm_zmq.send_multipart([comm_id, message])
        self.comm_stream.flush(zmq.POLLOUT)

    def send_scene(self,
                   websocket: Optional[WebSocketHandler] = None,
                   comm_id: Optional[bytes] = None) -> None:
        """ TODO: Write documentation.
        """
        if websocket is not None:
            super().send_scene(websocket)
        elif comm_id is not None:
            for node in walk(self.tree):
                if node.object is not None:
                    self.forward_to_comm(comm_id, node.object)
                for prop in node.properties:
                    self.forward_to_comm(comm_id, prop)
                if node.transform is not None:
                    self.forward_to_comm(comm_id, node.transform)


# ======================================================

def _meshcat_server(info: Dict[str, str], verbose: bool) -> None:
    """Meshcat server daemon, using in/out argument to get the zmq url instead
    of reading stdout as it was.
    """
    # pylint: disable=consider-using-with
    # Redirect both stdout and stderr to devnull if not verbose
    if not verbose:
        devnull = open(os.devnull, 'w')
        sys.stdin = sys.stderr = devnull

    # See https://bugs.python.org/issue37373
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy())

    # Do not catch signal interrupt automatically, to avoid
    # killing meshcat server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create new asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bridge = ZMQWebSocketIpythonBridge()
    info['zmq_url'] = bridge.zmq_url
    info['web_url'] = bridge.web_url
    info['comm_url'] = bridge.comm_url
    bridge.run()

    if not verbose:
        devnull.close()


def start_meshcat_server(verbose: bool = False
                         ) -> Tuple[multiprocessing.Process, str, str, str]:
    """Run meshcat server in background using multiprocessing Process.
    """
    with multiprocessing.managers.SyncManager() as manager:
        info = manager.dict()
        server = multiprocessing.Process(
            target=_meshcat_server, args=(info, verbose), daemon=True)
        server.start()

        # Wait for the process to finish initialization
        while 'comm_url' not in info.keys():
            pass
        zmq_url, web_url, comm_url = \
            info['zmq_url'], info['web_url'], info['comm_url']

    return server, zmq_url, web_url, comm_url


def start_meshcat_server_standalone() -> None:
    """ TODO: Write documentation.
    """
    argparse.ArgumentParser(description=(
        "Serve the Jiminy MeshCat HTML files and listen for ZeroMQ commands"))

    server, zmq_url, web_url, comm_url = start_meshcat_server(True)
    print(zmq_url)
    print(web_url)
    print(comm_url)

    try:
        server.join()
    except KeyboardInterrupt:
        server.terminate()
        server.join(timeout=0.5)
        try:
            proc = psutil.Process(server.pid)
            proc.send_signal(signal.SIGKILL)
        except psutil.NoSuchProcess:
            pass
