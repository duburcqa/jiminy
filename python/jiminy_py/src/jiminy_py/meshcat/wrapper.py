import atexit
from contextlib import redirect_stdout

import zmq

import meshcat

from .recorder import MeshcatRecorder


class MeshcatWrapper:
    def __init__(self, zmq_url):
        with redirect_stdout(None):
            self.gui = meshcat.Visualizer(zmq_url)
        self.__zmq_socket = self.gui.window.zmq_socket
        self.recorder = MeshcatRecorder(self.gui.url())
        atexit.register(self.close)

    def __del__(self):
        self.close()

    def close(self):
        self.recorder.release()

    def wait(self, require_client=False):
        if require_client:
            self.gui.wait()
        self.__zmq_socket.send(b"ready")
        self.__zmq_socket.recv().decode("utf-8")

    def start_recording(self, fps, width, height):
        if not self.recorder.is_open:
            self.recorder.open()
            self.wait(require_client=True)
        self.recorder.start_video_recording(fps, width, height)

    def stop_recording(self, path):
        self.recorder.stop_and_save_video(path)

    def add_frame(self):
        self.recorder.add_video_frame()

    def capture_frame(self, width=None, height=None):
        if not self.recorder.is_open:
            self.recorder.open()
            self.wait(require_client=True)
        return self.recorder.capture_frame(width, height)
