import os
import sys
import signal
import atexit
import asyncio
import subprocess
import multiprocessing
from ctypes import c_char_p, c_bool, c_int
from contextlib import redirect_stderr

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
        "--ignore-certificate-errors",
        "--disable-infobars",
        "--disable-breakpad",
        "--disable-setuid-sandbox",
        "--proxy-server='direct://'",
        "--proxy-bypass-list=*",
        "--enable-webgl",
        "--disable-frame-rate-limit",
        "--disable-gpu-vsync"]
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

    # don't forget to close browser process
    def _close_process(*args, **kwargs) -> None:
        if not self.chromeClosed:
            self._loop.run_until_complete(self.killChrome())
    atexit.register(_close_process)
    if self.handleSIGTERM:
        signal.signal(signal.SIGTERM, _close_process)
    if not sys.platform.startswith('win'):
        if self.handleSIGHUP:
            signal.signal(signal.SIGHUP, _close_process)

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

def capture_frame(client, width, height):
    """
    @brief    Send a javascript command to the hidden browser to
              capture frame, then wait for it (since it is async).
    """
    async def capture_frame_async(client):
        nonlocal width, height
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
    loop = asyncio.get_event_loop()
    img_data = loop.run_until_complete(capture_frame_async(client))
    return img_data

def meshcat_recorder(meshcat_url,
                     take_snapshot_shm,
                     img_data_html_shm,
                     width_shm,
                     height_shm):
    # Do not catch signal interrupt automatically, to avoid killing meshcat
    # server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Open a Meshcat client in background in a hidden chrome browser instance
    session = HTMLSession()
    client = session.get(meshcat_url)
    client.html.render(keep_page=True)

    # Stop rendering loop since it is irrelevant in his case, because
    # capture_frame is already doing the job.
    async def stop_animation_async(client):
        return await client.html.page.evaluate("""
            () => {
                stop_animate();
            }
        """)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(stop_animation_async(client))

    with open(os.devnull, 'w') as f:
        with redirect_stderr(f):
            while True:
                # Wait for request to take a screenshot
                if take_snapshot_shm.value:
                    img_data_html_shm.value = capture_frame(
                        client, width_shm.value, height_shm.value)
                    take_snapshot_shm.value = False

def mgr_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def start_meshcat_recorder(meshcat_url):
    """
    @brief    Run meshcat server in background using multiprocessing
              Process to enable parallel asyncio loop execution, which
              is necessary to support recording in Jupyter notebook.
    """
    manager = multiprocessing.managers.SyncManager()
    manager.start(mgr_init)
    recorder_shm = {
        'take_snapshot': manager.Value(c_bool, False),
        'img_data_html': manager.Value(c_char_p, ""),
        'width': manager.Value(c_int, -1),
        'height': manager.Value(c_int, -1)
    }

    recorder = multiprocessing.Process(
        target=meshcat_recorder,
        args=(meshcat_url,
            recorder_shm['take_snapshot'],
            recorder_shm['img_data_html'],
            recorder_shm['width'],
            recorder_shm['height']),
        daemon=True)
    recorder.start()

    return recorder, manager, recorder_shm
