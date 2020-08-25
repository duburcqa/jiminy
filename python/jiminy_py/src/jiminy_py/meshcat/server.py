import os
import psutil
import signal
import asyncio
import umsgpack
import tornado.web
import multiprocessing
from contextlib import redirect_stderr

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

def make_app(self):
    return tornado.web.Application([
        (r"/static/?(.*)", MyFileHandler, {
            "default_path": VIEWER_ROOT,
            "fallback_path": os.path.dirname(__file__),
            "default_filename": "index.html"}),
        (r"/", WebSocketHandler, {"bridge": self})
    ])
ZMQWebSocketBridge.make_app = make_app

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
    self.bridge.websocket_messages.append(message)
    if len(self.bridge.websocket_messages) == len(self.bridge.websocket_pool):
        gathered_msg = ",".join(self.bridge.websocket_messages)
        self.bridge.zmq_socket.send(gathered_msg.encode("utf-8"))
        self.bridge.zmq_stream.flush()
WebSocketHandler.on_message = handle_web

def wait_for_websockets(self):
    if len(self.websocket_pool) > 0:
        self.zmq_socket.send(b"ok")
        self.zmq_stream.flush()
    else:
        self.ioloop.call_later(0.1, self.wait_for_websockets)
ZMQWebSocketBridge.wait_for_websockets = wait_for_websockets

handle_zmq_orig = ZMQWebSocketBridge.handle_zmq
def handle_zmq(self, frames):
    self.websocket_messages = []  # Used to gather websocket messages
    cmd = frames[0].decode("utf-8")
    if cmd == "meshes_loaded":
        if not self.websocket_pool:
            self.zmq_socket.send("".encode("utf-8"))
        for websocket in self.websocket_pool:
            websocket.write_message(umsgpack.packb({
                u"type": u"meshes_loaded"
            }), binary=True)
    else:
        handle_zmq_orig(self, frames)
ZMQWebSocketBridge.handle_zmq = handle_zmq

# ======================================================

def meshcat_server(info):
    """
    @brief   Meshcat server deamon, using in/out argument to get the
             zmq url instead of reading stdout as it was.
    """
    # Do not catch signal interrupt automatically, to avoid
    # killing meshcat server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create asyncio event loop if not already existing
    asyncio.get_event_loop()

    with open(os.devnull, 'w') as f:
        with redirect_stderr(f):
            bridge = ZMQWebSocketBridge()
            info['zmq_url'] = bridge.zmq_url
            info['web_url'] = bridge.web_url
            bridge.run()

def start_meshcat_server():
    """
    @brief    Run meshcat server in background using multiprocessing
              Process to enable monkey patching and proper interprocess
              communication through a manager.
    """
    manager = multiprocessing.Manager()
    info = manager.dict()
    server = multiprocessing.Process(
        target=meshcat_server, args=(info,), daemon=True)
    server.start()

    # Wait for the process to finish initialization
    while not info:
        pass
    zmq_url, web_url = info['zmq_url'], info['web_url']
    manager.shutdown()

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
