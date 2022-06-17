import os
import urllib
import base64
import atexit
import asyncio
import logging
import pathlib
import umsgpack
import threading
import tornado.ioloop
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Sequence, Dict, Any

import zmq
from zmq.eventloop.zmqstream import ZMQStream

import meshcat

from .utilities import interactive_mode
from .server import start_meshcat_server
from .recorder import MeshcatRecorder


if interactive_mode() >= 3:
    # Google colab is using an older version of ipykernel (4.10), which is
    # not compatible with >= 5.0. The new API is more flexible and enable
    # to process only the relevant messages because every incoming messages
    # is first added in a priority queue waiting for being processed. Thus,
    # it is possible to process part of those messages without altering the
    # other ones. It is not possible with the old API since every incoming
    # message must be either processed right after flushing, or discarded.
    # Emulating or restore the queue would be possible theoretically but it
    # is tricky to do it properly, so instead every message is process
    # without distinction.
    import ipykernel
    ipykernel_version_major = int(ipykernel.__version__[0])
    if ipykernel_version_major < 6:
        from ipykernel.kernelbase import SHELL_PRIORITY
    elif ipykernel_version_major > 6:
        logging.warning(
            "ipykernel version 7 detected. The viewer works optimally with "
            " ipykernel 5 or 6. Revert to old version in case of issues.")

    class CommProcessor:
        """Re-implementation of ipykernel.kernelbase.do_one_iteration to only
        handle comm messages on the spot, and put back in the stack the other
        ones.

        Calling 'do_one_iteration' messes up with kernel `msg_queue`. Some
        messages will be processed too soon, which is likely to corrupt the
        kernel state. This method only processes comm messages to avoid such
        side effects.
        """
        def __init__(self):
            from IPython import get_ipython
            self.__kernel = get_ipython().kernel
            self._is_colab = (interactive_mode() == 4)
            self.qsize_old = 0
            self.is_running = False

        def __call__(self) -> None:
            """Check once if there is pending comm related event in the shell
            stream message priority queue.
            """
            # Guard to avoid running this method several times in parallel
            if self.is_running:
                return
            self.is_running = True

            # Flush every IN messages on shell_stream only.
            # Note that it is a faster implementation of `ZMQStream.flush()`
            # to only handle incoming messages. It reduces the computation from
            # about 15us to 15ns.
            # https://github.com/zeromq/pyzmq/blob/e424f83ceb0856204c96b1abac93a1cfe205df4a/zmq/eventloop/zmqstream.py#L313
            if ipykernel_version_major > 5:
                shell_stream = self.__kernel.shell_stream
            else:
                shell_stream = self.__kernel.shell_streams[0]
            shell_stream.flush(zmq.POLLIN)

            # One must go through all the messages to keep them in order
            for _ in range(self.__kernel.msg_queue.qsize()):
                try:
                    *priority, t, dispatch, args = \
                        self.__kernel.msg_queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Just in case the queue has been emptied in the meantime,
                    # which should never happen in practice
                    break
                if not priority or priority[0] <= SHELL_PRIORITY:
                    # New message: reading message without deserializing its
                    # content at this point for efficiency.
                    try:
                        idents, msg = self.__kernel.session.feed_identities(
                            args[-1], copy=False)
                        msg = self.__kernel.session.deserialize(
                            msg, content=False, copy=False)
                    except ValueError:
                        # Corrupted message. Skipping it.
                        msg = None
                else:
                    # Do not spend time analyzing messages already rejected
                    msg = None

                if msg is not None and \
                        msg['header']['msg_type'].startswith('comm_'):
                    # Extract comm type and handler
                    comm_type = msg['header']['msg_type']

                    # Extract message content
                    content = self.__kernel.session.unpack(msg['content'])
                    data = content.get('data', '')

                    # Analyzing comm message to determine whether it is related
                    # to meshcat and process it on the spot.
                    is_meschat_comm_request = False
                    if self._is_colab and comm_type == 'comm_close':
                        # All comm_close messages are processed because Google
                        # Colab API does not expose sending data on close to
                        # specify that it is a meshcat-related message.
                        is_meschat_comm_request = True
                    if isinstance(data, str) and data.startswith('meshcat:'):
                        # Comm message related to meshcat. Processing it right
                        # now and moving to the next message without putting it
                        # back into the queue.
                        is_meschat_comm_request = True

                    # Process the request if necessary
                    if is_meschat_comm_request:
                        # Unpack message content
                        msg['content'] = content

                        # Backup original kernel parent before hijacking
                        original_parent = (
                            self.__kernel._parent_ident,
                            self.__kernel.get_parent()
                            if hasattr(self.__kernel, "get_parent")
                            else self.__kernel._parent_header)

                        # Note that it is necessary to set the kernel parent
                        # and publish idle status when processing the message
                        # because otherwise google colab will never acknowledge
                        # that the comm connection has been established. Still,
                        # it is important to restore the original parent
                        # afterward, otherwise the current cell will never
                        # return to idle state on display, although the kernel
                        # is not actually stuck and other cells can be
                        # evaluated properly. Yet, trying to stop it will crash
                        # the kernel unsurprisingly. Conversely, setting the
                        # parent on jupyter is interrupting the cell completely
                        # and definitively...
                        if self._is_colab:
                            self.__kernel.set_parent(idents, msg)
                            self.__kernel._publish_status('busy')
                        comm_handler = self.__kernel.shell_handlers[comm_type]
                        comm_handler(shell_stream, idents, msg)
                        if self._is_colab:
                            self.__kernel._publish_status('idle')
                            self.__kernel.set_parent(*original_parent)
                        shell_stream.flush(zmq.POLLOUT)
                        continue

                # The message is not related to meshcat comm, so putting it
                # back in the queue at the "end of the queue", ie just at the
                # right place: after the next unchecked messages, after the
                # other messages already put back in the queue, but before the
                # next one to go the same way.
                # Note that its priority is also lowered, so that the next time
                # it can be directly forwarded without analyzing its content,
                # since every shell messages have SHELL_PRIORITY by default.
                # Note that ipykernel 6 removed priority feature.
                if priority:
                    self.__kernel.msg_queue.put_nowait(
                        (SHELL_PRIORITY + 1, t, dispatch, args))
                else:
                    self.__kernel.msg_queue.put_nowait((t, dispatch, args))
            self.qsize_old = self.__kernel.msg_queue.qsize()

            # Ensure the event loop wakes up
            self.__kernel.io_loop.add_callback(lambda: None)

            # Disable guard
            self.is_running = False

    # Start comm hijacking
    process_kernel_comm = CommProcessor()


class CommManager:
    def __new__(cls, *args: Any, **kwargs: Any) -> "CommManager":
        self = super().__new__(cls)
        self.__ioloop = None
        self.__comm_socket = None
        self.__comm_stream = None
        self.__thread = None
        self.__kernel = None
        return self

    def __init__(self, comm_url: str):
        from IPython import get_ipython

        def forward_comm_thread():
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.__ioloop = tornado.ioloop.IOLoop()

            # Make sure the communication are processed at least once every
            # seconds for the redirection of comm msg related to watchdogs.
            def background_watchdog() -> None:
                # Process comm messages if any
                if self.__comm_stream is not None:
                    self.__comm_stream.flush(zmq.POLLIN)
                process_kernel_comm()

                # Re-schedule the method
                if self.__ioloop is not None:
                    self.__ioloop.call_later(1.0, background_watchdog)
            background_watchdog()

            # Start comm socket
            context = zmq.Context()
            self.__comm_socket = context.socket(zmq.XREQ)
            self.__comm_socket.connect(comm_url)
            self.__comm_stream = ZMQStream(self.__comm_socket, self.__ioloop)
            self.__comm_stream.on_recv(self.__forward_to_ipykernel)
            self.__ioloop.start()

            # Stop running socket
            self.__ioloop.close()
            self.__ioloop = None
            self.__comm_stream.close(linger=5)
            self.__comm_stream = None
            self.__comm_socket.close(linger=5)
            self.__comm_socket = None

        self.__thread = threading.Thread(target=forward_comm_thread)
        self.__thread.daemon = True
        self.__thread.start()

        self.__kernel = get_ipython().kernel
        self.__kernel.comm_manager.register_target(
            'meshcat', self.__comm_register)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if 'meshcat' in self.__kernel.comm_manager.targets:
            self.__kernel.comm_manager.unregister_target(
                'meshcat', self.__comm_register)
        if self.__ioloop is not None:
            self.__ioloop.stop()
        if self.__thread is not None:
            self.__thread.join()
            self.__thread = None

    def __forward_to_ipykernel(self, frames: Sequence[bytes]) -> None:
        comm_id, *cmd = frames
        comm_id = comm_id.decode()
        try:
            comm = self.__kernel.comm_manager.comms[comm_id]
            comm.send(buffers=cmd)
        except KeyError:
            # The comm has probably been closed without the server knowing.
            # Sending the notification to the server to consider it as such.
            self.__comm_socket.send(f"close:{comm_id}".encode())

    def __comm_register(self,
                        comm: 'ipykernel.comm.Comm',  # noqa
                        msg: Dict[str, Any]) -> None:
        # There is a major limitation of using `comm.on_msg` callback
        # mechanism: if the main thread is already busy for some reason, for
        # instance waiting for a reply from the server ZMQ socket, then
        # `comm.on_msg` will NOT triggered automatically. It is only triggered
        # automatically once every other tasks has been process. The workaround
        # is to interleave blocking code with call of `kernel.do_one_iteration`
        # or `await kernel.process_one(wait=True)`. See Stackoverflow for ref:
        # https://stackoverflow.com/a/63666477/4820605
        @comm.on_msg
        def _on_msg(msg: Dict[str, Any]) -> None:
            data = msg['content']['data'][8:]  # Remove 'meshcat:' header
            self.__comm_socket.send(f"data:{comm.comm_id}:{data}".encode())
            self.__comm_stream.flush(zmq.POLLOUT)

        @comm.on_close
        def _close(evt: Any) -> None:
            self.__comm_socket.send(f"close:{comm.comm_id}".encode())
            self.__comm_stream.flush(zmq.POLLOUT)

        self.__comm_socket.send(f"open:{comm.comm_id}".encode())
        self.__comm_stream.flush(zmq.POLLOUT)


class MeshcatWrapper:
    def __new__(cls, *args: Any, **kwargs: Any) -> "MeshcatWrapper":
        self = super().__new__(cls)
        self.server_proc = None
        self.recorder = None
        self.comm_manager = None
        self.__zmq_socket = None
        return self

    def __init__(self,
                 zmq_url: Optional[str] = None,
                 comm_url: Optional[str] = None):
        # Launch a custom meshcat server if necessary
        must_launch_server = zmq_url is None
        if must_launch_server:
            self.server_proc, zmq_url, _, comm_url = start_meshcat_server(
                verbose=False)

        # Connect to the meshcat server
        with open(os.devnull, 'w') as stdout, redirect_stdout(stdout):
            with open(os.devnull, 'w') as stderr, redirect_stderr(stderr):
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
        if must_launch_server and interactive_mode() >= 3:
            self.comm_manager = CommManager(comm_url)

        # Make sure the server is properly closed
        atexit.register(self.close)

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self.__zmq_socket is not None:
            self.__zmq_socket.close()
            self.__zmq_socket = None
        if self.comm_manager is not None:
            self.comm_manager.close()
            self.comm_manager = None
        if self.recorder is not None:
            self.recorder.release()
            self.recorder = None

    def wait(self, require_client: bool = False) -> str:
        if require_client:
            # Calling the original `wait` method must be avoided since it is
            # blocking. Here we are waiting for a new comm to connect. Always
            # perform a single `do_one_iteration`, just in case there is
            # already comm waiting in the queue to be registered, but it should
            # not be necessary.
            self.__zmq_socket.send(b"wait")
            if self.comm_manager is None:
                self.__zmq_socket.recv()
            else:
                while True:
                    try:
                        # Try first, just in case there is already a comm for
                        # websocket available.
                        self.__zmq_socket.recv(flags=zmq.NOBLOCK)
                        break
                    except zmq.error.ZMQError:
                        # No websocket nor comm connection available at this
                        # point. Fetching new incoming messages and retrying.
                        # By doing this, opening a websocket or comm should
                        # be enough to successfully recv the acknowledgement.
                        process_kernel_comm()

        # Process every pending messages
        if self.comm_manager is not None:
            qsize_old = -1
            while qsize_old != process_kernel_comm.qsize_old:
                process_kernel_comm()
                qsize_old = process_kernel_comm.qsize_old

        # Send 'ready' request and wait for reply. Note that while a single zmq
        # reply is expected whatever the number of comms. New opening/closing
        # connection while awaiting for 'ready' acknowledgement is handled by
        # the server unless closed unexpectedly without notice. Therefore, the
        # total number of comm messages received may be smaller than the number
        # of comms currently registered. It is necessary to check for a reply
        # of the server periodically, and the number of responses corresponds
        # to the actual number of comms.
        self.__zmq_socket.send(b"ready")
        if self.comm_manager is not None:
            while True:
                process_kernel_comm()
                try:
                    msg = self.__zmq_socket.recv(flags=zmq.NOBLOCK)
                    return msg.decode("utf-8")
                except zmq.error.ZMQError:
                    pass
        return self.__zmq_socket.recv().decode("utf-8")

    def set_legend_item(self, uniq_id: str, color: str, text: str) -> None:
        self.__zmq_socket.send_multipart([
            b"set_property",      # Frontend command. Used by Python zmq server
            b"",                  # Tree path. Empty path means root
            umsgpack.packb({      # Backend command. Used by javascript
                "type": "legend",
                "id": uniq_id,   # Unique identifier of updated legend item
                "text": text,    # Any text message support by HTML5
                "color": color   # "rgba(0, 0, 0, 0.0)" and "black" supported
            })
        ])
        self.__zmq_socket.recv()  # Receive acknowledgement

    def remove_legend_item(self, uniq_id: str) -> None:
        self.__zmq_socket.send_multipart([
            b"set_property",
            b"",
            umsgpack.packb({
                u"type": "legend",
                u"id": uniq_id,   # Unique identifier of legend item to remove
                u"text": ""       # Empty message means delete the item, if any
            })
        ])
        self.__zmq_socket.recv()

    def set_watermark(self,
                      img_fullpath: str,
                      width: int,
                      height: int) -> None:
        # Handle file format
        url = urllib.parse.urlparse(img_fullpath)
        if all([url.scheme in ["http", "https"], url.netloc, url.path]):
            img_data = img_fullpath
        else:
            # Determine image format
            file_ext = pathlib.Path(img_fullpath).suffix
            if file_ext == ".png":
                img_format = "png"
            elif file_ext in (".jpeg", ".jpg"):
                img_format = "jpg"
            elif file_ext == ".svg":
                img_format = "svg+xml"
            else:
                raise ValueError(
                    f"Format {file_ext} not supported. It must be either "
                    "'.png', '.jpeg' or 'svg'.")

            # Convert image to base64
            with open(img_fullpath, "rb") as img_file:
                img_raw = base64.b64encode(img_file.read()).decode('utf-8')
            img_data = f"data:image/{img_format};base64,{img_raw}"

        # Send ZMQ request to acknowledge reply
        self.__zmq_socket.send_multipart([
            b"set_property",
            b"",
            umsgpack.packb({
                u"type": "watermark",
                u"data": img_data,
                u"width": width,
                u"height": height
            })
        ])
        self.__zmq_socket.recv()

    def remove_watermark(self) -> None:
        self.__zmq_socket.send_multipart([
            b"set_property",
            b"",
            umsgpack.packb({
                u"type": "watermark",
                u"data": ""   # Empty string means delete the watermark, if any
            })
        ])
        self.__zmq_socket.recv()

    def start_recording(self, fps: float, width: int, height: int) -> None:
        if not self.recorder.is_open:
            self.recorder.open()
            self.wait(require_client=True)
        self.recorder.start_video_recording(fps, width, height)

    def stop_recording(self, path: str) -> None:
        self.recorder.stop_and_save_video(path)

    def add_frame(self) -> None:
        self.recorder.add_video_frame()

    def capture_frame(self,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> str:
        if self.recorder.is_open:
            self.wait(require_client=False)
        else:
            self.recorder.open()
            self.wait(require_client=True)
        return self.recorder.capture_frame(width, height)
