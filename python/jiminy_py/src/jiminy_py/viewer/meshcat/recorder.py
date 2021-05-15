import os
import sys
import time
import logging
import signal
import psutil
import pathlib
import asyncio
import subprocess
import multiprocessing
import multiprocessing.managers
from pathlib import Path
from ctypes import c_char_p
from contextlib import redirect_stderr
from typing import Awaitable, Any, Optional

from requests_html import HTMLSession, HTMLResponse

from .utilities import interactive_mode


if interactive_mode() == 2:
    # Must overload 'chromium_executable' for Google Colaboratory to
    # the native browser instead: "/usr/lib/chromium-browser/chromium-browser".
    # Note that the downside is that chrome must be installed manually.
    import pyppeteer.chromium_downloader
    pyppeteer.chromium_downloader.chromium_executable = \
        lambda: Path("/usr/lib/chromium-browser/chromium-browser")
    if not pyppeteer.chromium_downloader.check_chromium():
        logging.warning(
            "Chrome must be installed manually on Google Colab. It must be "
            "done using '!apt install chromium-chromedriver'.")
elif not sys.platform.startswith('win'):
    # Must use a recent release that supports webgl rendering with hardware
    # acceleration. It speeds up rendering at least by a factor 5 using on
    # a midrange dedicated GPU.
    os.environ['PYPPETEER_CHROMIUM_REVISION'] = '801225'


# ==================== Monkey-patch pyppeteer ============================

from pyppeteer.errors import NetworkError
from pyppeteer.connection import (
    Connection, logger, logger_connection, logger_session)
from pyppeteer.browser import Browser
from pyppeteer.launcher import Launcher, get_ws_endpoint


# Disable pyppeteer logger(s)
logger.disabled = True
logger_connection.disabled = True
logger_session.disabled = True


# Make sure raise SIGINT does not kill chrome pyppeteer browser backend
# automatically, so that it allows a closing handle to be manually registered.
# Note that only "egl" webgl renderer supports hardware acceleration in
# headless mode without X server on Linux. Moreover, headless mode with
# hardware acceleration enables seems to crash Chrome on Windows. If so, it can
# be disable by adding the extra command '"--disable-gpu"'.
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
        "--ignore-gpu-blacklist",
        "--ignore-certificate-errors",
        "--disable-infobars",
        "--disable-breakpad",
        "--disable-setuid-sandbox",
        "--proxy-server='direct://'",
        "--proxy-bypass-list=*"]

    if not self.dumpio:
        options['stdout'] = subprocess.DEVNULL
        options['stderr'] = subprocess.DEVNULL
    if sys.platform.startswith('win'):
        startupflags = subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.version_info >= (3, 7):
            startupflags |= (subprocess.DETACHED_PROCESS |
                             subprocess.HIGH_PRIORITY_CLASS)
        self.proc = subprocess.Popen(
            cmd, **options, creationflags=startupflags, shell=False)
    else:
        def proc_setup():
            os.setpgrp()
            os.nice(20)

        self.proc = subprocess.Popen(
            cmd, **options, preexec_fn=proc_setup, shell=False)

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
Launcher.launch = launch  # noqa


# ============ Asynchronous Javascript routines ============

async def capture_frame_async(client: HTMLResponse,
                              width: int,
                              height: int) -> Awaitable[Any]:
    """Send a javascript command to the hidden browser to capture frame, then
    wait for it (since it is async).
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


async def start_video_recording_async(client: HTMLResponse,
                                      fps: int,
                                      width: int,
                                      height: int) -> Awaitable[None]:
    await client.html.page.setViewport(
            {'width': width, 'height': height})
    await client.html.page.evaluate(f"""
        () => {{
            viewer.animator.capturer = new WebMWriter({{
                quality: 0.99999,  // Lossless codex VP8L is not supported
                frameRate: {fps}
            }});
        }}
    """)


async def add_video_frame_async(client: HTMLResponse) -> Awaitable[None]:
    await client.html.page.evaluate("""
        () => {
            captureFrameAndWidgets(viewer).then(function(canvas) {
                viewer.animator.capturer.addFrame(canvas);
            });
        }
    """)


async def stop_and_save_video_async(client: HTMLResponse,
                                    path: str) -> Awaitable[None]:
    directory = os.path.dirname(path)
    filename = os.path.splitext(os.path.basename(path))[0]
    await client.html.page._client.send(
        'Page.setDownloadBehavior',
        {'behavior': 'allow', 'downloadPath': directory})
    await client.html.page.evaluate(f"""
        () => {{
            viewer.animator.capturer.complete().then(function(blob) {{
                const a = document.createElement('a');
                const url = URL.createObjectURL(blob);
                a.href = url;
                a.download = "{filename}";
                a.click();
            }});
        }}
    """)


# ============ Meshcat recorder multiprocessing worker ============

def meshcat_recorder(meshcat_url: str,
                     request_shm: multiprocessing.Value,
                     message_shm: multiprocessing.Value) -> None:
    # Raise exception is Python is not installed properly
    if sys.platform.startswith('win') and "WindowsApps" in sys.executable:
        raise RuntimeError(
            "Python installed from Microsoft Store is not compatible with "
            "pyppeteer auto-install backend chromium procedure. Please "
            "re-install Python manually to be able to use Meshcat recorder.")

    # Do not catch signal interrupt automatically, to avoid killing meshcat
    # server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create new asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Open a Meshcat client in background in a hidden chrome browser instance
    session = HTMLSession()
    client = session.get(meshcat_url)
    client.html.render(keep_page=True)
    message_shm.value = f"{session._browser.process.pid}"

    # Stop the animation loop by default, since it is not relevant for
    # recording only
    async def stop_animate_async(client):
        await client.html.page.evaluate("""
            () => {
                stop_animate();
            }
        """)
    loop.run_until_complete(stop_animate_async(client))

    # Infinite loop, waiting for requests
    try:
        while request_shm.value != "quit":
            request = request_shm.value
            if request != "":
                args = map(str.strip, message_shm.value.split("|"))
                if request == "take_snapshot":
                    width, height = map(int, args)
                    coro = capture_frame_async(client, width, height)
                elif request == "start_record":
                    fps, width, height = map(int, args)
                    coro = start_video_recording_async(
                        client, fps, width, height)
                elif request == "add_frame":
                    coro = add_video_frame_async(client)
                elif request == "stop_and_save_record":
                    (path,) = args
                    coro = stop_and_save_video_async(client, path)
                else:
                    continue
                try:
                    output = loop.run_until_complete(coro)
                    if output is not None:
                        message_shm.value = output
                    else:
                        message_shm.value = ""
                except Exception as e:
                    message_shm.value = str(e)
                    request_shm.value = "quit"
                else:
                    request_shm.value = ""
    except (ConnectionError, NetworkError):
        pass
    with open(os.devnull, 'w') as stderr, redirect_stderr(stderr):
        session.close()
    try:
        message_shm.value = ""
        request_shm.value = ""
    except ConnectionError:
        pass


# ============ Meshcat recorder client ============

def manager_process_startup():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class MeshcatRecorder:
    """Run meshcat server in background using multiprocessing Process to enable
    parallel asyncio loop execution, which is necessary to support recording in
    Jupyter notebook.
    """
    def __init__(self, url: str):
        self.is_open = False
        self.is_recording = False
        self.url = url
        self.__manager = None
        self.__shm = None
        self.proc = None
        self.__browser_pid = None

    def open(self) -> None:
        self.__manager = multiprocessing.managers.SyncManager()
        self.__manager.start(manager_process_startup)

        self.__shm = {
            'request': self.__manager.Value(c_char_p, ""),
            'message': self.__manager.Value(c_char_p, "")
        }

        self.proc = multiprocessing.Process(
            target=meshcat_recorder,
            args=(self.url, self.__shm['request'], self.__shm['message']),
            daemon=True)
        self.proc.start()

        while self.__shm['message'].value == "":
            pass
        self.__browser_pid = int(self.__shm['message'].value)
        self.__shm['message'].value = ""

        self.is_open = True

    def __del__(self) -> None:
        self.release()

    def release(self) -> None:
        if self.__shm is not None:
            if self.proc.is_alive():
                try:
                    self._send_request(request="quit", timeout=0.5)
                except Exception:
                    # This method must not fail under any circumstances
                    pass
            self.__shm = None
        if self.proc is not None:
            self.proc.terminate()
            self.proc = None
        if self.__browser_pid is not None:
            try:
                psutil.Process(self.__browser_pid).kill()
                os.waitpid(self.__browser_pid, 0)
                os.waitpid(os.getpid(), 0)
            except (psutil.NoSuchProcess, ChildProcessError):
                pass
            self.__browser_pid = None
        if self.__manager is not None:
            self.__manager.shutdown()
            self.__manager = None
        self.is_open = False

    def _send_request(self,
                      request: str,
                      message: Optional[str] = None,
                      timeout: float = 15.0) -> None:
        if not self.is_open:
            raise RuntimeError(
                "Meshcat recorder is not open. Impossible to send requests.")
        if message is not None:
            self.__shm['message'].value = message
        else:
            self.__shm['message'].value = ""
        self.__shm['request'].value = request
        timeout += time.time()
        while self.__shm['request'].value != "":
            if time.time() > timeout:
                self.release()
                raise RuntimeError("Timeout.")
            elif not self.proc.is_alive():
                self.release()
                raise RuntimeError(
                    "Backend browser has encountered an unrecoverable "
                    "error: ", self.__shm['message'].value)

    def capture_frame(self,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> str:
        self._send_request(
            "take_snapshot", message=f"{width or -1}|{height or -1}")
        return self.__shm['message'].value

    def start_video_recording(self,
                              fps: float,
                              width: int,
                              height: int) -> None:
        self._send_request(
            "start_record", message=f"{fps}|{width}|{height}", timeout=10.0)
        self.is_recording = True

    def add_video_frame(self) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "No video being recorded at the moment. Please start "
                "recording before adding frames.")
        self._send_request("add_frame")

    def stop_and_save_video(self, path: str) -> bool:
        def file_available(path):
            if not os.path.exists(path):
                return False
            for proc in psutil.process_iter():
                try:
                    if 'chrome' in proc.name():
                        for item in proc.open_files():
                            if path == item.path:
                                return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # The process ended before examining its files
                    pass
            return True

        if not self.is_recording:
            raise RuntimeError(
                "No video being recorded at the moment. Please start "
                "recording and add frames before saving.")
        if "|" in path:
            raise ValueError(
                "'|' character is not supported in video export path.")
        path = os.path.abspath(pathlib.Path(path).with_suffix('.webm'))
        if os.path.exists(path):
            os.remove(path)
        self._send_request("stop_and_save_record", message=path, timeout=30.0)
        self.is_recording = False
        while not file_available(path):
            pass
