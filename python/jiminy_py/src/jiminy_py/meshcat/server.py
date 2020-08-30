import os
import psutil
import signal
import asyncio
import umsgpack
import tornado.web
import multiprocessing
from contextlib import redirect_stderr

import zmq
from zmq.eventloop.zmqstream import ZMQStream

from meshcat.servers.tree import walk, find_node
from meshcat.servers.zmqserver import (
    VIEWER_ROOT, ZMQWebSocketBridge, WebSocketHandler)

# ================ Monkey-patch =======================

# Add support of cross-origin connection.
# It is useful to execute custom javascript commands within
# a Jupyter Notebook, and it is not an actual security flaw
# for local servers since they are not accessible from the
# outside anyway.
WebSocketHandler.check_origin = lambda self, origin: True

# Override the default html page to disable auto-update of
# three js "controls" of the camera, so that it can be moved
# programmatically in any position, without any constraint, as
# long as the user is not moving it manually using the mouse.
class MyFileHandler(tornado.web.StaticFileHandler):
    def initialize(self, default_path, default_filename, fallback_path):
        self.default_path = os.path.abspath(default_path)
        self.default_filename = default_filename
        self.fallback_path = os.path.abspath(fallback_path)
        super().initialize(self.default_path, self.default_filename)

    def set_extra_headers(self, path):
        self.set_header('Cache-Control',
                        'no-store, no-cache, must-revalidate, max-age=0')

    def validate_absolute_path(self, root, absolute_path):
        if os.path.isdir(absolute_path):
            if not self.request.path.endswith("/"):
                self.redirect(self.request.path + "/", permanent=True)
                return None
            absolute_path = os.path.join(absolute_path, self.default_filename)
            return self.validate_absolute_path(root, absolute_path)
        if os.path.exists(absolute_path) and \
                os.path.basename(absolute_path) != self.default_filename:
            return super().validate_absolute_path(root, absolute_path)
        return os.path.join(self.fallback_path, absolute_path[(len(root)+1):])

# Implement bidirectional communication because zmq and the
# websockets by gathering and forward messages received from
# the websockets to zmq. Note that there is currently no way
# to identify the client associated to each reply, but it is
# usually not a big deal, since the same answers is usual
# expected from each of them. Comma is used as a delimiter.
#
# It also fixes flushing issue when 'handle_zmq' is not directly
# responsible for sending a message through the zmq socket.
def handle_web(self, message):
    self.bridge.websocket_msg.append(message)
    if len(self.bridge.websocket_msg) == len(self.bridge.websocket_pool) and \
            len(self.bridge.comm_msg) == len(self.bridge.comm_pool):
        gathered_msg = ",".join(
            self.bridge.websocket_msg + self.bridge.comm_msg)
        self.bridge.zmq_socket.send(gathered_msg.encode("utf-8"))
        self.bridge.zmq_stream.flush()
        self.bridge.websocket_msg, self.bridge.comm_msg = [], []
WebSocketHandler.on_message = handle_web

class ZMQWebSocketIpythonBridge(ZMQWebSocketBridge):
    def __init__(self, zmq_url=None, host="127.0.0.1", port=None):
        super().__init__(zmq_url, host, port)
        self.comm_pool = []
        self.comm_url = "tcp://127.0.0.1:6678" # TODO : Check if port is available
        self.zmq_socket_comm = self.context.socket(zmq.XREQ)
        self.zmq_socket_comm.bind(self.comm_url)
        self.zmq_stream_comm = ZMQStream(self.zmq_socket_comm)
        self.zmq_stream_comm.on_recv(self.handle_comm)
        self.websocket_msg = []  # Used to gather websocket messages
        self.comm_msg = []  # Used to gather comm messages

    def make_app(self):
        return tornado.web.Application([
            (r"/static/?(.*)", MyFileHandler, {
                "default_path": VIEWER_ROOT,
                "fallback_path": os.path.dirname(__file__),
                "default_filename": "index.html"}),
            (r"/", WebSocketHandler, {"bridge": self})
        ])

    def wait_for_websockets(self):
        if len(self.websocket_pool) > 0 or len(self.comm_pool) > 0:
            self.zmq_socket.send(b"ok")
            self.zmq_stream.flush()
        else:
            self.ioloop.call_later(0.1, self.wait_for_websockets)

    def handle_zmq(self, frames):
        cmd = frames[0].decode("utf-8")
        if cmd == "ready":
            if not self.websocket_pool and not self.comm_pool: #TODO: Disable temporarily to avoid hanging
                self.zmq_socket.send(b"")
            msg = umsgpack.packb({"type": "ready"})
            for websocket in self.websocket_pool:
                websocket.write_message(msg, binary=True)
            for comm_id in self.comm_pool:
                self.forward_to_comm(comm_id, msg)
        else:
            super().handle_zmq(frames)

    def handle_comm(self, frames):
        cmd = frames[0].decode("utf-8")
        if cmd.startswith("open:"):
            comm_id = f"{cmd.split(':', 1)[1]}"
            self.send_scene(comm_id=comm_id)
            self.comm_pool.append(comm_id)
        elif cmd.startswith("close:"):
            comm_id = f"{cmd.split(':', 1)[1]}"
            self.comm_pool.remove(comm_id)
        elif cmd.startswith("data:"):
            message = f"{cmd.split(':', 2)[2]}"
            self.comm_msg.append(message)
            if len(self.websocket_msg) == len(self.websocket_pool) and \
                    len(self.comm_msg) == len(self.comm_pool):
                gathered_msg = ",".join(
                    self.websocket_msg + self.comm_msg)
                self.zmq_socket.send(gathered_msg.encode("utf-8"))
                self.zmq_stream.flush()
                self.websocket_msg, self.comm_msg = [], []
            self.comm_msg = []

    def forward_to_websockets(self, frames):
        # Check if the objects are still available in cache
        cmd, path, data = frames
        cache_hit = (cmd == "set_object" and
                     find_node(self.tree, path).object and
                     find_node(self.tree, path).object == data)
        if cache_hit:
            return
        super().forward_to_websockets(frames)
        _, _, data = frames
        for comm_id in self.comm_pool:
            self.forward_to_comm(comm_id, data)

    def forward_to_comm(self, comm_id, message):
        self.zmq_socket_comm.send(comm_id.encode() + data)

    def send_scene(self, websocket=None, comm_id=None):
        if websocket is not None:
            super().send_scene(websocket)
        elif comm_id is not None:
            for node in walk(self.tree):
                if node.object is not None:
                    self.forward_to_comm(comm_id, node.object)
                if node.transform is not None:
                    self.forward_to_comm(comm_id, node.transform)

# ======================================================

def meshcat_server(info):
    """
    @brief   Meshcat server deamon, using in/out argument to get the
             zmq url instead of reading stdout as it was.
    """
    # Do not catch signal interrupt automatically, to avoid
    # killing meshcat server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create new asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with open(os.devnull, 'w') as f:
        with redirect_stderr(f):
            bridge = ZMQWebSocketIpythonBridge()
            info['comm_url'] = bridge.comm_url
            info['zmq_url'] = bridge.zmq_url
            info['web_url'] = bridge.web_url
            bridge.run()

def start_meshcat_server():
    """
    @brief    Run meshcat server in background using multiprocessing Process.
    """
    manager = multiprocessing.Manager()
    info = manager.dict()
    server = multiprocessing.Process(
        target=meshcat_server, args=(info,), daemon=True)
    server.start()

    # Wait for the process to finish initialization
    while not info:
        pass
    comm_url, zmq_url, web_url = \
        info['comm_url'], info['zmq_url'], info['web_url']
    manager.shutdown()

    # Create ZMQ Socket to Ipython Kernel bridge.
    # Note that it must be done on host, not on remote, because kernel
    # communication are not available in subprocess (roughly speaking).
    # It is an one directional communication. The aim is only to redirect
    # messages send to the websockets to the kernel communication 'meshcat'.
    def forward_to_ipython(messages):
        from IPython import get_ipython
        comm_pool = get_ipython().kernel.comm_manager.comms
        for message in messages:
            comm_id = message[:32].decode()  # comm_id is always 32 bits
            comm_pool[comm_id].send(buffers=[message[32:]])

    context = zmq.Context.instance()
    zmq_socket_comm = context.socket(zmq.XREQ)
    zmq_socket_comm.connect(comm_url)
    zmq_stream_comm = ZMQStream(zmq_socket_comm)
    zmq_stream_comm.on_recv(forward_to_ipython)

    # Implement bi-directional communication using 'comm.on_msg'. ZMQ
    # ROUTER/ROUTER protocol supports it, even though it is hacky. A double
    # socket is used to avoid altering too much the original implementation.
    # ROUTER/ROUTER enables to spend and receive as many messages as desired
    # consecutively without replying systematically between them.
    try:
        kernel = get_ipython().kernel
    except (NameError, AttributeError):
        pass  # No backend Ipython kernel available. Not listening for incoming connections.
    else:
        context = zmq.Context()
        zmq_socket_comm = context.socket(zmq.XREQ)
        zmq_socket_comm.connect(comm_url)
        def comm_register(comm, msg):
            nonlocal zmq_socket_comm

            @comm.on_msg
            def _on_msg(msg):
                nonlocal zmq_socket_comm
                data = msg['content']['data']
                zmq_socket_comm.send(f"data:{comm.comm_id}:{data}".encode())

            @comm.on_close
            def _close(evt):
                nonlocal zmq_socket_comm
                zmq_socket_comm.send(f"close:{comm.comm_id}".encode())

            zmq_socket_comm.send(f"open:{comm.comm_id}".encode())
        kernel.comm_manager.register_target('meshcat', comm_register)

    return server, zmq_url, web_url

def start_meshcat_server_standalone():
    import argparse
    argparse.ArgumentParser(
        description="Serve the Jiminy MeshCat HTML files and listen for ZeroMQ commands")

    server, zmq_url, web_url = start_meshcat_server()
    print(zmq_url)
    print(web_url)

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
