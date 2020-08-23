import os
import signal
import asyncio
import multiprocessing
from ctypes import c_char_p, c_bool, c_int
from contextlib import redirect_stderr

from requests_html import BaseSession, HTMLSession

# ================ Monkey-patch =======================

# Overwrite pyppeteer headless browser backend options.
async def browser(self):
    browser_args = {
        'headless': True,
        'args': self.__browser_args
    }
    if not hasattr(self, "_browser"):
        self._browser = await pyppeteer.launch(
            ignoreHTTPSErrors=not(self.verify), **browser_args)
    print("Nice job !")
    return self._browser
BaseSession.browser = property(browser)

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
    # Do not catch signal interrupt automatically, to avoid
    # killing meshcat server and stopping Jupyter notebook cell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    session = HTMLSession()
    client = session.get(meshcat_url)
    client.html.render(keep_page=True)

    # with open(os.devnull, 'w') as f:
    #     with redirect_stderr(f):
    while True:
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

    return recorder, recorder_shm
