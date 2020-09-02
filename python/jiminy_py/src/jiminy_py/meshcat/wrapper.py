import atexit
import asyncio
import logging
import threading
import tornado.ioloop
from contextlib import redirect_stdout

import zmq
from zmq.eventloop.zmqstream import ZMQStream

import meshcat
from .server import start_meshcat_server

from .recorder import MeshcatRecorder


def is_notebook():
    """
    @brief Determine whether Python is running inside a Notebook or not.
    """
    from IPython import get_ipython
    shell = get_ipython().__class__.__module__
    if shell == 'ipykernel.zmqshell':
        return 1   # Jupyter notebook or qtconsole. Impossible to discriminate easily without costly psutil inspection of the running process...
    elif shell == 'IPython.terminal.interactiveshell':
        return 0   # Terminal running IPython
    elif shell.startswith('google.colab.'):
        return 2   # Google Colaboratory
    elif shell == '__builtin__':
        return 0   # Terminal running Python
    else:
        return 0   # Unidentified type


if is_notebook() == 1:
    # The IO message rate limit has already been increased to 1e6 on Google
    # Colab, so no need to throw this warning.
    logging.warning(
        "You may experience some lags while replaying a simulation.\n"\
         "Consider increasing the IO message rate limit by adding the "\
         "extra argument '--NotebookApp.iopub_msg_rate_limit=100000' when "\
         "executing 'jupyter notebook'.")

if is_notebook():
    # Google colab is using an older version of ipykernel (4.10), which is
    # not compatible with >= 5.0. The new API is more flexible and enable
    # to process only the relevant messages because every incoming messages
    # is first added in a priority queue waiting for being processed. Thus,
    # it is possible to process part of those messages without altering the
    # other ones. It is not possible with the old API since every incoming
    # message must be ever processed just after flushing, or discarded.
    # Emulating or restore the queue would be possible theoretically but it
    # is tricky to do it properly, so instead every message is process
    # without distinction.
    import ipykernel
    from pkg_resources import parse_version as version
    if version(ipykernel.__version__) >= version("5.0"):
        import tornado.gen
        from ipykernel.kernelbase import SHELL_PRIORITY
    else:
        logging.warning(
            "Old ipykernel version < 5.0 detected. Please do not schedule "\
            "other cells for execution while the viewer is busy otherwise "\
            "it will be not executed properly.\nUpdate to a newer version "\
            "if possible to avoid such limitation.")

    class CommProcessor:
        """
        @brief     Re-implementation of ipykernel.kernelbase.do_one_iteration
                   to only handle comm messages on the spot, and put back in
                   the stack the other ones.

        @details   Calling 'do_one_iteration' messes up with kernel
                   'msg_queue'. Some messages will be processed too soon,
                   which is likely to corrupt the kernel state. This method
                   only processes comm messages to avoid such side effects.
        """

        def __init__(self):
            self.__kernel = get_ipython().kernel
            self.__old_api = version(ipykernel.__version__) < version("5.0")
            if self.__old_api:
                logging.warning("Pre/post kernel handler hooks must be "\
                    "disable for the old ipykernel API to enable fetching"\
                    "shell messages from child threads.")
                self.__kernel.post_handler_hook = lambda : None
                self.__kernel.pre_handler_hook = lambda : None
            self.qsize_old = 0

        def __call__(self, unsafe=False):
            """
            @brief      Check once if there is pending comm related event in
                        the shell stream message priority queue.

            @param[in]  unsafe     Whether or not to assume check if the number
                                   of pending message has changed is enough. It
                                   makes the evaluation much faster but flawed.
            """
            # Flush every IN messages on shell_stream only
            # Note that it is a faster implementation of ZMQStream.flush
            # to only handle incoming messages. It reduces the computation
            # from about 15us to 15ns.
            # https://github.com/zeromq/pyzmq/blob/e424f83ceb0856204c96b1abac93a1cfe205df4a/zmq/eventloop/zmqstream.py#L313
            shell_stream = self.__kernel.shell_streams[0]
            shell_stream.poller.register(shell_stream.socket, zmq.POLLIN)
            events = shell_stream.poller.poll(0)
            while events:
                _, event = events[0]
                if event:
                    shell_stream._handle_recv()
                    shell_stream.poller.register(
                        shell_stream.socket, zmq.POLLIN)
                    events = shell_stream.poller.poll(0)

            if self.__old_api:
                return  # The messages have already been processed...

            qsize = self.__kernel.msg_queue.qsize()
            if unsafe and qsize == self.qsize_old:
                # The number of queued messages in the queue has not changed
                # since it last time it has been checked. Assuming those
                # messages are the same has before and returning earlier.
                return

            # One must go through all the messages to keep them in order
            for _ in range(qsize):
                priority, t, dispatch, args = \
                    self.__kernel.msg_queue.get_nowait()
                if priority <= SHELL_PRIORITY:
                    _, msg = self.__kernel.session.feed_identities(
                        args[1], copy=False)
                    msg = self.__kernel.session.deserialize(
                        msg, content=False, copy=False)
                else:
                    # Do not spend time analyzing already rejected message
                    msg = None
                if msg is None or not 'comm_' in msg['header']['msg_type']:
                    # The message is not related to comm, so putting it back in
                    # the queue after lowering its priority so that it is send
                    # at the "end of the queue", ie just at the right place:
                    # after the next unchecked messages, after the other
                    # messages already put back in the queue, but before the
                    # next one to go the same way. Note that every shell
                    # messages have SHELL_PRIORITY by default.
                    self.__kernel.msg_queue.put_nowait(
                        (SHELL_PRIORITY + 1, t, dispatch, args))
                else:
                    # Comm message. Processing it right now.
                    tornado.gen.maybe_future(dispatch(*args))
            self.qsize_old = self.__kernel.msg_queue.qsize()

    process_kernel_comm = CommProcessor()

    # Monkey-patch meshcat ViewerWindow 'send' method to process queued comm
    # messages. Otherwise, new opening comm will not be detected soon enough.
    _send_orig = meshcat.visualizer.ViewerWindow.send
    def _send(self, command):
        _send_orig(self, command)
        # Check on new comm related messages. Unsafe in enabled to avoid
        # potentially significant overhead. At this point several safe should
        # have been executed, so it is much less likely than comm messages
        # will slip through the net. Besides, missing messages at this point
        # is not blocking, because here we are not waiting for it to continue.
        process_kernel_comm(unsafe=True)
    meshcat.visualizer.ViewerWindow.send = _send


class CommManager:
    def __init__(self, comm_url):
        self.n_comm = 0
        self.n_message = 0

        def forward_comm_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ioloop = tornado.ioloop.IOLoop()
            context = zmq.Context()
            self.__comm_socket = context.socket(zmq.XREQ)
            self.__comm_socket.connect(comm_url)
            self.__comm_stream = ZMQStream(self.__comm_socket)
            self.__comm_stream.on_recv(self.__forward_to_ipython)
            ioloop.start()

        self.__thread = threading.Thread(target=forward_comm_thread)
        self.__thread.daemon = True
        self.__thread.start()

        self.__kernel = get_ipython().kernel
        self.__kernel.comm_manager.register_target(
            'meshcat', self.__comm_register)

    def __del__(self):
        self.close()

    def close(self):
        self.n_comm = 0
        self.n_message = 0
        self.__kernel.comm_manager.unregister_target(
            'meshcat', self.__comm_register)
        self.__thread._stop()
        self.__comm_stream.close(linger=5)
        self.__comm_socket.close(linger=5)

    def __forward_to_ipython(self, frames):
        comm_id, cmd = frames  # There must be always two parts each messages
        comm_id = comm_id.decode()
        try:
            comm = self.__kernel.comm_manager.comms[comm_id]
        except KeyError:
            # The comm has probably been closed without the server knowing.
            # Sending the notification to the server to consider it as such.
            self.__comm_socket.send(f"close:{comm_id}".encode())
        else:
            comm.send(buffers=[cmd])

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
            self.n_message += 1
            data = msg['content']['data']
            self.__comm_socket.send(f"data:{comm.comm_id}:{data}".encode())

        @comm.on_close
        def _close(evt):
            self.n_comm -= 1
            self.__comm_socket.send(f"close:{comm.comm_id}".encode())

        self.n_comm += 1
        self.__comm_socket.send(f"open:{comm.comm_id}".encode())


class MeshcatWrapper:
    def __init__(self, zmq_url=None, comm_url=None):
        # Launch a custom meshcat server if necessary
        must_launch_server = zmq_url is None
        self.server_proc = None
        if must_launch_server:
            self.server_proc, zmq_url, _, comm_url = start_meshcat_server()

        # Connect to the meshcat server
        with redirect_stdout(None):
            self.gui = meshcat.Visualizer(zmq_url)
        self.__zmq_socket = self.gui.window.zmq_socket

        # Create a backend recorder. It is not fully initialized to reduce
        # overhead when not used, which is way more usual than the contrary.
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
        self.comm_manager = None
        if must_launch_server and is_notebook():
            self.comm_manager = CommManager(comm_url)

        # Make sure the server is properly closed
        atexit.register(self.close)

    def __del__(self):
        self.close()

    def close(self):
        if self.comm_manager is not None:
            self.comm_manager.close()
        self.recorder.release()

    def wait(self, require_client=False):
        if require_client:
            # Calling the original 'wait' method must be avoided
            # since it is blocking. Here we are waiting for a
            # new comm to connect. Always perform a single
            # 'do_one_iteration', just in case there is already
            # comm waiting in the queue to be registered, but it
            # should not be necessary.
            self.__zmq_socket.send(b"wait")
            if self.comm_manager is None:
                self.__zmq_socket.recv()
            else:
                while True:
                    try:
                        # First try, just in case there is already a comm for
                        # websocket available.
                        self.__zmq_socket.recv(flags=zmq.NOBLOCK)
                        break
                    except zmq.error.ZMQError:
                        # No websocket nor comm connection available at this
                        # point. Fetching new incoming messages and retrying.
                        # By doing this, opening a websocket or comm should
                        # be enough to successfully recv the acknowledgement.
                        process_kernel_comm()

        self.__zmq_socket.send(b"ready")
        if self.comm_manager is not None:
            self.comm_manager.n_message = 0
            while self.comm_manager.n_message < self.comm_manager.n_comm:
                process_kernel_comm()
        return self.__zmq_socket.recv().decode("utf-8")

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
