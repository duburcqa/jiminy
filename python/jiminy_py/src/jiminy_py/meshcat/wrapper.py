import atexit
import asyncio
from heapq import heappop
from contextlib import redirect_stdout

import zmq
from zmq.eventloop.zmqstream import ZMQStream

import meshcat

from .recorder import MeshcatRecorder


class MeshcatWrapper:
    def __init__(self, zmq_url, comm_url):
        with redirect_stdout(None):
            self.gui = meshcat.Visualizer(zmq_url)
        self.__zmq_socket = self.gui.window.zmq_socket
        self.recorder = MeshcatRecorder(self.gui.url())

        # Create ZMQ socket from/to Ipython Kernel bridge to forward
        # communications the Javascript frontend in a notebook cell to the ZMQ
        # socket of the Meshcat server in both directions. Note that it must be
        # done on host, not on remote, because kernel communication are not
        # available not official supported. Forwarding communications from
        # Ipython kernel to ZMQ socket is implemented using 'comm.on_msg'
        # handle, which is managed by the  kernel and synchronized with the
        # main asyncio loop. The other direction is handled using dedicated ZMQ
        # sockets using ROUTER/ROUTER protocol. It also sending and receiving
        # any number of messages from both sides without systematic reply, much
        # like PUB/SUB + SUB/PUB double sockets. Note that ROUTER/ROUTER
        # communication is supported by the standard but not encouraged. It has
        # been chosen to add extra ROUTER/ROUTER sockets instead of replacing
        # the original ones to avoid altering too much the original
        # implementation of Meshcat.
        self.n_comm = 0
        self.__n_message = 0
        try:
            self.__kernel = get_ipython().kernel
        except (NameError, AttributeError):
            pass  # No backend Ipython kernel available. Not listening for incoming connections.
        else:
            context = zmq.Context().instance()
            self.__comm_socket = context.socket(zmq.XREQ)
            self.__comm_socket.connect(comm_url)
            self.__comm_stream = ZMQStream(self.__comm_socket)
            self.__comm_stream.on_recv(self.__forward_to_ipython)
            self.__kernel.comm_manager.register_target(
                'meshcat', self.__comm_register)

        atexit.register(self.close)

    def __del__(self):
        self.close()

    def close(self):
        self.n_comm = 0
        self.__n_message = 0
        self.recorder.release()
        if self.__kernel is not None:
            self.__kernel.comm_manager.unregister_target(
                'meshcat', self.__comm_register)
        self.__comm_stream.close(linger=5)
        self.__comm_socket.close(linger=5)

    def __forward_to_ipython(self, messages):
        from IPython import get_ipython
        comm_pool = get_ipython().kernel.comm_manager.comms
        for message in messages:
            comm_id = message[:32].decode()  # comm_id is always 32 bits
            comm_pool[comm_id].send(buffers=[message[32:]])

    def __comm_register(self, comm, msg):
        # There is a major limitation of using 'comm.on_msg' callback
        # mechanism: if the main thread is already busy for some reason, for
        # instance waiting for a reply from the server ZMQ socket, then
        # 'comm.on_msg' will NOT triggered automatically. It is only triggered
        # automatically once every other tacks has been process. The workaround
        # is to interleave blocking code with call of 'kernel.do_one_iteration'
        # or 'await kernel.process_one(wait=True)'. See Stackoverflow for ref.
        # https://stackoverflow.com/questions/63651823/direct-communication-between-javascript-in-jupyter-and-server-via-ipython-kernel/63666477#63666477
        @comm.on_msg
        def _on_msg(msg):
            self.__n_message += 1
            data = msg['content']['data']  # TODO: Check compatibility with Google Colab
            self.__comm_socket.send(f"data:{comm.comm_id}:{data}".encode())

        @comm.on_close
        def _close(evt):
            self.n_comm -= 1
            self.__comm_socket.send(f"close:{comm.comm_id}".encode())

        self.n_comm += 1
        self.__comm_socket.send(f"open:{comm.comm_id}".encode())

    @staticmethod
    def __recv(coro):
        loop = asyncio.get_event_loop()
        f = asyncio.ensure_future(coro)
        while not f.done():
            # Extract from the implementation of 'run_until_complete' in
            # 'nested_asyncio' package. Check Github for reference
            # https://github.com/erdewit/nest_asyncio/blob/43467067b87b20325c4cbaaf78af8eff16ecf847/nest_asyncio.py#L80
            now = loop.time()
            ready = loop._ready
            scheduled = loop._scheduled
            while scheduled and scheduled[0]._cancelled:
                heappop(scheduled)
            timeout = 0 if ready or loop._stopping \
                else min(max(0, scheduled[0]._when - now), 10) if scheduled \
                else None
            event_list = loop._selector.select(timeout)
            loop._process_events(event_list)
            while scheduled and scheduled[0]._when < now + loop._clock_resolution:
                handle = heappop(scheduled)
                ready.append(handle)
            while ready:
                handle = ready.popleft()
                if not handle._cancelled:
                    handle._run()
            handle = None
        return f.result()

    def wait(self, require_client=False):
        if require_client:
            self.gui.wait()

        async def wait_async():
            while self.__n_message < self.n_comm:
                await self.__kernel.process_one(wait=True)
            return self.__zmq_socket.recv().decode("utf-8")

        self.__n_message = 0
        self.__zmq_socket.send(b"ready")
        self.__recv(wait_async())

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
