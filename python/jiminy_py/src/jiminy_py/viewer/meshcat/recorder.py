
""" TODO: Write documentation.
"""
import asyncio
import signal
import subprocess
import multiprocessing
import multiprocessing.managers
import os
import time
from contextlib import redirect_stderr
from ctypes import c_char_p
from pathlib import Path
from typing import Any, Optional, Dict

import psutil

from playwright.sync_api import sync_playwright, Page, Error, ViewportSize
from playwright._impl._driver import compute_driver_executable, get_driver_env


PLAYWRIGHT_DOWNLOAD_TIMEOUT = 180.0  # 3min to download browser (~130Mo)
PLAYWRIGHT_START_TIMEOUT = 40000.0   # 40s

WINDOW_SIZE_DEFAULT = (600, 600)


# ============ Javascript routines ============

def _stop_animate(client: Page) -> None:
    """ TODO: Write documentation.
    """
    client.evaluate("""
        () => {
            stop_animate();
        }
    """)


def _capture_frame(client: Page, width: int, height: int) -> Any:
    """Send a javascript command to the hidden browser to capture frame, then
    wait for it (since it is async).
    """
    # Assert(s) for type checker
    assert client.viewport_size is not None

    _width = client.viewport_size['width']
    _height = client.viewport_size['height']
    if not width > 0:
        width = _width
    if not height > 0:
        height = _height
    if _width != width or _height != height:
        client.set_viewport_size({"width": width, "height": height})
    return client.evaluate("""
        () => {
            return viewer.capture_image();
        }
    """)


def _start_video_recording(
        client: Page, fps: int, width: int, height: int) -> None:
    """ TODO: Write documentation.
    """
    client.set_viewport_size({'width': width, 'height': height})
    client.evaluate(f"""
        () => {{
            viewer.animator.capturer = new WebMWriter({{
                quality: 0.99,  // Lossless codex VP8L is not supported
                frameRate: {fps}
            }});
        }}
    """)


def _add_video_frame(client: Page) -> None:
    """ TODO: Write documentation.
    """
    client.evaluate("""
        () => {
            captureFrameAndWidgets(viewer).then(function(canvas) {
                viewer.animator.capturer.addFrame(canvas);
            });
        }
    """)


def _stop_and_save_video(client: Page, path: Path) -> None:
    """ TODO: Write documentation.
    """
    # Start waiting for the download
    with client.expect_download() as download_info:
        client.evaluate(f"""
            () => {{
                viewer.animator.capturer.complete().then(function(blob) {{
                    const a = document.createElement('a');
                    const url = URL.createObjectURL(blob);
                    a.href = url;
                    a.download = "{path.name}";
                    a.click();
                }});
            }}
        """)
    # Save downloaded file
    download_info.value.save_as(path)


# ============ Meshcat recorder multiprocessing worker ============

def meshcat_recorder(meshcat_url: str,
                     request_shm: multiprocessing.managers.ValueProxy,
                     message_shm: multiprocessing.managers.ValueProxy) -> None:
    """ TODO: Write documentation.
    """
    # Do not catch signal interrupt automatically, to avoid killing meshcat
    # server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create new asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Download browser if necessary
    try:
        subprocess.run(
            [str(compute_driver_executable()), "install", "chromium"],
            timeout=PLAYWRIGHT_DOWNLOAD_TIMEOUT,
            env=get_driver_env(),
            check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Impossible to download browser.") from e

    # Create headless browser in background and connect to Meshcat
    browser = None
    playwright = sync_playwright().start()
    try:
        with open(os.devnull, 'w') as stderr, redirect_stderr(stderr):
            browser = playwright.chromium.launch(
                headless=True,
                handle_sigint=False,
                args=[
                    "--enable-webgl",
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan,UseSkiaRenderer",
                    "--use-vulkan=swiftshader",
                    "--use-angle=vulkan",
                    "--use-gl=egl",
                    # "--use-gl=swiftshader",
                    "--disable-gpu-vsync",
                    "--ignore-gpu-blacklist"
                ],
                timeout=PLAYWRIGHT_START_TIMEOUT)
        client = browser.new_page(
            viewport=ViewportSize(
                width=WINDOW_SIZE_DEFAULT[0],
                height=WINDOW_SIZE_DEFAULT[1]),
            java_script_enabled=True,
            accept_downloads=True)
        client.goto(
            meshcat_url, wait_until="load", timeout=PLAYWRIGHT_START_TIMEOUT)
        message_shm.value = str(
            browser._impl_obj._browser_type._connection._transport._proc.pid)
    except Error as e:
        request_shm.value = "quit"
        message_shm.value = str(e)

    # Stop the animation loop by default, since it is not relevant for
    # recording only
    if request_shm.value != "quit":
        _stop_animate(client)

    # Infinite loop, waiting for requests
    try:
        while request_shm.value != "quit":
            request = request_shm.value
            if request != "":
                # pylint: disable=broad-exception-caught
                args = map(str.strip, message_shm.value.split("|"))
                output = None
                try:
                    if request == "take_snapshot":
                        width, height = map(int, args)
                        output = _capture_frame(client, width, height)
                    elif request == "start_record":
                        fps, width, height = map(int, args)
                        _start_video_recording(client, fps, width, height)
                    elif request == "add_frame":
                        _add_video_frame(client)
                    elif request == "stop_and_save_record":
                        (path,) = args
                        _stop_and_save_video(client, Path(path))
                    else:
                        continue
                    message_shm.value = output if output is not None else ""
                    request_shm.value = ""
                except Exception as e:
                    message_shm.value = str(e)
                    break
    except (ConnectionError, Error):
        pass
    if browser is not None:
        with open(os.devnull, 'w') as stderr, redirect_stderr(stderr):
            try:
                browser.close()
            except Error:
                pass


# ============ Meshcat recorder client ============

def _manager_process_startup() -> None:
    """Required for Windows and OS X support, which use spawning instead of
    forking to create subprocesses, requiring passing pickle-compliant
    objects, and therefore prohibiting the use of native lambda functions.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class MeshcatRecorder:
    """Run meshcat server in background using multiprocessing Process to enable
    parallel asyncio loop execution, which is necessary to support recording in
    Jupyter notebook.
    """
    def __init__(self, url: str):
        self.url = url
        self.is_open = False
        self.is_recording = False
        self.proc: Optional[multiprocessing.Process] = None
        self.__manager: Optional[multiprocessing.managers.BaseManager] = None
        self.__shm: Optional[
            Dict[str, multiprocessing.managers.ValueProxy]] = None
        self.__browser_pid: Optional[int] = None

    def open(self) -> None:
        """ TODO: Write documentation.
        """
        # pylint: disable=consider-using-with
        self.__manager = multiprocessing.managers.SyncManager()
        self.__manager.start(_manager_process_startup)

        self.__shm = {
            'request': self.__manager.Value(c_char_p, ""),
            'message': self.__manager.Value(c_char_p, "")
        }

        self.proc = multiprocessing.Process(
            target=meshcat_recorder,
            args=(self.url, self.__shm['request'], self.__shm['message']),
            daemon=True)
        self.proc.start()

        timeout = PLAYWRIGHT_DOWNLOAD_TIMEOUT + PLAYWRIGHT_START_TIMEOUT * 1e-3
        time_start, time_waiting = time.time(), 0.0
        while self.__shm['message'].value == "":
            time_waiting = time.time() - time_start
            if time_waiting > timeout:
                break
        if self.__shm['request'].value == "quit":
            msg = "Impossible to start browser in background"
            raise RuntimeError(f"{msg}:\n    {self.__shm['message'].value}")
        if time_waiting > timeout:
            raise RuntimeError("Backend browser not responding.")
        self.__browser_pid = int(self.__shm['message'].value)
        self.__shm['message'].value = ""

        self.is_open = True

    def __del__(self) -> None:
        self.release()

    def release(self) -> None:
        """ TODO: Write documentation.
        """
        if hasattr(self, "__shm"):
            if hasattr(self, "proc") and self.proc is not None:
                if self.proc.is_alive():
                    # pylint: disable=broad-exception-caught
                    try:
                        self._send_request(request="quit", timeout=2.0)
                    except Exception:
                        # This method must not fail under any circumstances
                        pass
            self.__shm = None
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.terminate()
            self.proc = None
        if hasattr(self, "__browser_pid") and self.__browser_pid is not None:
            try:
                psutil.Process(self.__browser_pid).kill()
                os.waitpid(self.__browser_pid, 0)
                os.waitpid(os.getpid(), 0)
            except (psutil.NoSuchProcess, ChildProcessError):
                pass
            self.__browser_pid = None
        if hasattr(self, "__manager"):
            if hasattr(self.__manager, "shutdown"):
                # Assert(s) for type checker
                assert self.__manager is not None

                # `SyncManager.shutdown` method is only available once started
                self.__manager.shutdown()
            self.__manager = None
        self.is_open = False

    def _send_request(self,
                      request: str,
                      message: Optional[str] = None,
                      timeout: Optional[float] = None) -> None:
        """ TODO: Write documentation.
        """
        if timeout is None:
            timeout = PLAYWRIGHT_START_TIMEOUT * 1e-3
        if not self.is_open:
            raise RuntimeError(
                "Meshcat recorder is not open. Impossible to send requests.")

        # Assert(s) for type checker
        assert self.__shm is not None and self.proc is not None

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
            if not self.proc.is_alive():
                err = self.__shm['message'].value
                self.release()
                raise RuntimeError(
                    "Backend browser has encountered an unrecoverable "
                    f"error:\n{err}")

    def capture_frame(self,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> str:
        """ TODO: Write documentation.
        """
        # Assert(s) for type checker
        assert self.__shm is not None

        self._send_request(
            "take_snapshot", message=f"{width or -1}|{height or -1}")
        return self.__shm['message'].value

    def start_video_recording(self,
                              fps: float,
                              width: int,
                              height: int) -> None:
        """ TODO: Write documentation.
        """
        self._send_request(
            "start_record", message=f"{fps}|{width}|{height}", timeout=10.0)
        self.is_recording = True

    def add_video_frame(self) -> None:
        """ TODO: Write documentation.
        """
        if not self.is_recording:
            raise RuntimeError(
                "No video being recorded at the moment. Please start "
                "recording before adding frames.")
        self._send_request("add_frame")

    def stop_and_save_video(self, path: str) -> None:
        """ TODO: Write documentation.
        """
        if not self.is_recording:
            raise RuntimeError(
                "No video being recorded at the moment. Please start "
                "recording and add frames before saving.")
        if "|" in path:
            raise ValueError(
                "'|' character is not supported in video export path.")
        path = Path(path).with_suffix('.webm').absolute()
        if path.exists():
            path.unlink()
        self._send_request(
            "stop_and_save_record", message=str(path))
        self.is_recording = False
