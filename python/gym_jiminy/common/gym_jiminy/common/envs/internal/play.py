""" TODO: Write documentation.
"""
# pylint: disable=import-outside-toplevel,import-error

import os
import queue
import time
import threading
from typing import Optional, Callable, Any

from jiminy_py.viewer import sleep


class Getch:
    """Catch a single character from standard input before it echoes to the
    screen.
    """
    def __init__(self,
                 stop_event: Optional[threading.Event] = None,
                 max_rate: Optional[float] = None) -> None:
        """ TODO: Write documentation.
        """
        self.stop_event = stop_event
        self.max_rate = max_rate
        if os.name != 'nt':
            import sys
            import fcntl
            import termios
            self.fd = sys.stdin.fileno()
            self.oldterm = termios.tcgetattr(self.fd)
            newattr = termios.tcgetattr(self.fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSANOW, newattr)
            self.oldflags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            newflags = self.oldflags | os.O_NONBLOCK
            fcntl.fcntl(self.fd, fcntl.F_SETFL, newflags)

    def __del__(self) -> None:
        """ TODO: Write documentation.
        """
        if os.name != 'nt':
            import fcntl
            import termios
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.oldterm)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.oldflags)

    def __call__(self) -> str:
        """ TODO: Write documentation.
        """
        if os.name != 'nt':  # pylint: disable=no-else-return
            char = ''
            try:
                import sys
                import termios
                termios.tcflush(self.fd, termios.TCIFLUSH)
                while self.stop_event is None or \
                        not self.stop_event.is_set():
                    if self.max_rate is not None:
                        time.sleep(self.max_rate)
                    try:
                        char += sys.stdin.read(1)
                        if char and (char[:1] != '\x1b' or len(char) > 2):
                            break
                    except IOError:
                        pass
            except Exception:  # pylint: disable=broad-except
                pass
            return char
        else:
            import msvcrt
            while self.stop_event is None or \
                    not self.stop_event.is_set():
                if self.max_rate is not None:
                    time.sleep(self.max_rate)
                if msvcrt.kbhit():  # type: ignore[attr-defined]
                    return msvcrt.getch()  # type: ignore[attr-defined]
            return ''


def input_deamon(input_queue: queue.Queue,
                 stop_event: threading.Event,
                 exit_key: str,
                 max_rate: float) -> None:
    """ TODO: Write documentation.
    """
    char_to_arrow_mapping = {"\x1b[A": "Up",
                             "\x1b[B": "Down",
                             "\x1b[C": "Right",
                             "\x1b[D": "Left"}
    getch = Getch(stop_event, max_rate)
    while not stop_event.is_set():
        char = getch()
        if char in char_to_arrow_mapping.keys():
            char = char_to_arrow_mapping[char]
        if list(bytes(char.encode('utf-8'))) == [3]:
            char = exit_key
        input_queue.put(char)
    del getch


def loop_interactive(exit_key: str = 'k',
                     pause_key: str = 'p',
                     start_paused: bool = True,
                     max_rate: Optional[float] = 1e-3,
                     verbose: bool = True) -> Callable[
                         [Callable[..., bool]], Callable[..., None]]:
    """Create a wrapper responsible of calling a method periodically,
    while forwarding input keys using `key` keyword argument. It
    loops indefinitely until the forwarded method returns `True`, or the exist
    key is pressed.

    :param exit_key: Key to press to break the loop.
                     Optional: 'k' by default.
    :param pause_key: Key to press to pause the loop.
                      Optional: 'p' by default.
    :param start_paused: Whether or not to start in pause.
                         Optional: Enable by default.
    :param max_rate: Maximum rate of the loop. If slowing down the loop is
                     necessary, then busy loop is used instead of sleep for
                     maximum accurary.
                     Optional: 1e-3 s by default.
    :param verbose: Whether or not to display status messages.
                    Optional: Enable by default.
    """
    assert pause_key != exit_key, "Cannot use the same key for pause and exit."

    def wrap(func: Callable[..., bool]) -> Callable[..., None]:
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            nonlocal start_paused, pause_key, exit_key

            # Start keyboard input handling thread
            input_queue: queue.Queue = queue.Queue()
            stop_event = threading.Event()
            input_thread = threading.Thread(
                target=input_deamon,
                args=(input_queue, stop_event, exit_key, max_rate),
                daemon=True)
            input_thread.start()

            # Display status messages
            if verbose:
                print("Entering keyboard interactive mode. Pressed "
                      f"'{exit_key}' to exit or close window.")
            if pause_key:
                print(f"Press '{pause_key}' to start...")

            # Loop infinitly until termination is triggered
            key = None
            stop = False
            is_paused = start_paused
            try:
                while not stop:
                    # Get current time
                    t_init = time.time()

                    # Call wrapped function is already started
                    if not is_paused:
                        try:
                            stop = func(*args, **kwargs, key=key)
                        except KeyboardInterrupt:
                            stop = True
                        except Exception as e:  # pylint: disable=broad-except
                            print(str(e))
                            stop = True

                    # Sleep for a while if necessary, using busy loop only if
                    # already started to avoid unecessary cpu load.
                    if max_rate is not None and max_rate > 0.0:
                        dt = max(max_rate - (time.time() - t_init), 0.0)
                        if is_paused:
                            time.sleep(dt)
                        else:
                            sleep(dt)

                    # Look for new key pressed
                    key = None
                    while not input_queue.empty():
                        # Get new key
                        key = input_queue.get()

                        # Update stop flag if exit key pressed
                        if key == exit_key:
                            if verbose:
                                print("Exiting keyboard interactive mode.")
                            stop = True
                            key = None

                        # Update pause flag if pause key pressed
                        if key == pause_key:
                            if verbose:
                                if is_paused:
                                    print("Resume!")
                                else:
                                    print("Pause...")
                            is_paused = not is_paused
                            key = None

                        # Discard key if paused
                        if is_paused:
                            key = None
            except KeyboardInterrupt:
                pass
            finally:
                # Stop keyboard handling loop
                stop_event.set()
                time.sleep(max_rate + 0.01 if max_rate is not None else 0.01)
        return wrapped_func
    return wrap
