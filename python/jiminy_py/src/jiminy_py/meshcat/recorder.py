import os
import sys
import time
import signal
import atexit
import psutil
import pathlib
import asyncio
import subprocess
import multiprocessing
from ctypes import c_char_p, c_bool, c_int
from contextlib import redirect_stderr

# Must use a recent release that supports webgl rendering with hardware acceleration
os.environ['PYPPETEER_CHROMIUM_REVISION'] = '801225'

from pyppeteer.connection import Connection
from pyppeteer.browser import Browser
from pyppeteer.launcher import Launcher, get_ws_endpoint
from requests_html import HTMLSession

# ================ Monkey-patch =======================

# Make sure raise SIGINT does not kill chrome
# pyppeteer browser backend automatically, so that
# it allows a closing handle to be manually registered.
async def launch(self) -> Browser:
    """Start chrome process and return `Browser` object."""
    self.chromeClosed = False
    self.connection = None

    options = dict()
    options['env'] = self.env
    cmd = self.cmd + [
        "--enable-webgl",
        "--use-gl=egl",
        "--disable-frame-rate-limit",
        "--disable-gpu-vsync",
        "--ignore-certificate-errors",
        "--disable-infobars",
        "--disable-breakpad",
        "--disable-setuid-sandbox",
        "--proxy-server='direct://'",
        "--proxy-bypass-list=*"]
    if not self.dumpio:
        options['stdout'] = subprocess.PIPE
        options['stderr'] = subprocess.STDOUT
    if sys.platform.startswith('win'):
        startupflags = subprocess.DETACHED_PROCESS | \
            subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            cmd, **options, creationflags=startupflags, shell=False)
    else:
        self.proc = subprocess.Popen(
            cmd, **options, preexec_fn=os.setpgrp, shell=False)

    def _close_process(*args):
        try:
            if not self.chromeClosed:
                self._loop.run_until_complete(self.killChrome())
        except:
            self.proc.terminate()
            self.proc.join(timeout=0.5)
            try:
                proc_pid = self.proc.pid
                proc_raw = psutil.Process(proc_pid)
                proc_raw.send_signal(signal.SIGKILL)
                os.waitpid(proc_pid, 0)
                os.waitpid(os.getpid(), 0)
            except (psutil.NoSuchProcess, ChildProcessError):
                pass
            multiprocessing.active_children()
    signal.signal(signal.SIGTERM, _close_process)

    self.browserWSEndpoint = get_ws_endpoint(self.url)
    self.connection = Connection(self.browserWSEndpoint, self._loop)
    browser = await Browser.create(
        self.connection,
        [],
        self.ignoreHTTPSErrors,
        self.defaultViewport,
        self.proc,
        self.killChrome)
    await self.ensureInitialPage(browser)
    return browser
Launcher.launch = launch

# ======================================================

async def capture_frame_async(client, width, height):
    """
    @brief    Send a javascript command to the hidden browser to
            capture frame, then wait for it (since it is async).
    """
    _width = client.html.page.viewport['width']
    _height = client.html.page.viewport['height']
    if not width > 0:
        width = _width
    if not height > 0:
        height = _height
    if _width != width or _height != height:
        await client.html.page.setViewport(
            {'width': width, 'height': height})
    return await client.html.page.evaluate("""
        () => {
            return viewer.capture_image();
        }
    """)

async def start_video_recording_async(client, path, fps, width, height):
    directory = os.path.dirname(path)
    filename = os.path.splitext(os.path.basename(path))[0]
    await client.html.page.setViewport(
            {'width': width, 'height': height})
    await client.html.page._client.send('Page.setDownloadBehavior',
        {'behavior': 'allow', 'downloadPath': directory})
    await client.html.page.evaluate(f"""
        () => {{
            viewer.animator.capturer = CCapture(
                {{format: 'webm-mediarecorder', quality: 100.0, framerate: {fps}, name: '{filename}'}});
            viewer.animator.capturer.format = 'webm';
            viewer.animator.capturer.start();
        }}
    """)

async def add_video_frame_async(client):
    await client.html.page.evaluate("""
        () => {
            viewer.renderer.render(viewer.scene, viewer.camera);
            viewer.animator.capturer.capture(viewer.renderer.domElement);
        }
    """)

async def stop_and_save_video_async(client):
    return await client.html.page.evaluate("""
        () => {
            viewer.animator.capturer.stop();
            viewer.animator.capturer.save();
            viewer.animate();
        }
    """)

def meshcat_recorder(meshcat_url, request_shm, message_shm):
    # Do not catch signal interrupt automatically, to avoid killing meshcat
    # server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Open a Meshcat client in background in a hidden chrome browser instance
    session = HTMLSession()
    client = session.get(meshcat_url)
    client.html.render(keep_page=True)

    # Infinite loop, waiting for requests
    loop = asyncio.get_event_loop()
    with open(os.devnull, 'w') as f:
        with redirect_stderr(f):
            while request_shm.value != "quit": # (request := request_shm.value) != "quit":
                request = request_shm.value
                if request != "":
                    args = list(map(str.strip, message_shm.value.split(",")))
                    if request == "take_snapshot":
                        width, height = map(int, args)
                        coro = capture_frame_async(client, width, height)
                    elif request == "start_record":
                        path = args[0]
                        fps, width, height = map(int, args[1:])
                        coro = start_video_recording_async(
                            client, path, fps, width, height)
                    elif request == "add_frame":
                        coro = add_video_frame_async(client)
                    elif request == "stop_and_save_record":
                        coro = stop_and_save_video_async(client)
                    else:
                        continue
                    output = loop.run_until_complete(coro)
                    if output is not None:
                        message_shm.value = output
                    request_shm.value = ""
            request_shm.value = ""
    session.close()


class MeshcatRecorder:
    """
    @brief    Run meshcat server in background using multiprocessing
              Process to enable parallel asyncio loop execution, which
              is necessary to support recording in Jupyter notebook.
    """
    def __init__(self, url):
        self.is_open = False
        self.url = url
        self.__video_path = None
        self.__manager = None
        self.__shm = None
        self.proc = None

    def open(self):
        self.__manager = multiprocessing.managers.SyncManager()
        self.__manager.start(
            lambda : signal.signal(signal.SIGINT, signal.SIG_IGN))

        self.__shm = {
            'request': self.__manager.Value(c_char_p, ""),
            'message': self.__manager.Value(c_char_p, "")
        }

        self.proc = multiprocessing.Process(
            target=meshcat_recorder,
            args=(self.url, self.__shm['request'], self.__shm['message']),
            daemon=True)
        self.proc.start()

        self.is_open = True

    def __del__(self):
        self.release()

    def release(self):
        if self.__shm is not None:
            self._send_request(request="quit")
            self.__shm = None
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None
        if self.__manager is not None:
            self.__manager.shutdown()
            self.__manager = None
        self.is_open = False

    def _send_request(self, request, message=None, timeout=1000):
        if message is not None:
            self.__shm['message'].value = message
        else:
            self.__shm['message'].value = ""
        self.__shm['request'].value = request
        timeout += time.time()
        while self.__shm['request'].value != "":
            if time.time() > timeout:
                raise RuntimeError

    def capture_frame(self, width=None, height=None):
        self._send_request("take_snapshot",
            message=f"{width if width is not None else -1},"\
                    f"{height if height is not None else -1}")
        return self.__shm['message'].value

    def start_video_recording(self, path, fps, width, height):
        self.__video_path = pathlib.Path(path).with_suffix('.webm')
        self._send_request("start_record",
            message=f"{os.path.abspath(path)},{fps},{width},{height}")

    def add_video_frame(self):
        self._send_request("add_frame")

    def stop_and_save_video(self):
        self._send_request("stop_and_save_record")

