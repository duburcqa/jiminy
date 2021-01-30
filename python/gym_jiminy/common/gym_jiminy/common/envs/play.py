""" TODO: Write documentation.
"""
# pylint: disable=import-outside-toplevel,import-error

import os
import queue
import time
import threading
from typing import Optional, Callable, Any


class Getch:
    """Catch a single character from standard input before it echoes to the
    screen.
    """
    def __init__(self,
                 stop_event: Optional[threading.Event] = None) -> None:
        """ TODO: Write documentation.
        """
        self.stop_event = stop_event
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
                if msvcrt.kbhit():  # type: ignore[attr-defined]
                    return msvcrt.getch()  # type: ignore[attr-defined]
            return ''


def input_deamon(input_queue: queue.Queue,
                 stop_event: threading.Event,
                 exit_key: str) -> None:
    """ TODO: Write documentation.
    """
    char_to_arrow_mapping = {"\x1b[A": "Up",
                             "\x1b[B": "Down",
                             "\x1b[C": "Right",
                             "\x1b[D": "Left"}
    getch = Getch(stop_event)
    while not stop_event.is_set():
        char = getch()
        if char in char_to_arrow_mapping.keys():
            char = char_to_arrow_mapping[char]
        if list(bytes(char.encode('utf-8'))) == [3]:
            char = exit_key
        input_queue.put(char)
    del getch


def loop_interactive(press_key_to_start: bool = True,
                     exit_key: str = 'k') -> Callable:
    """ TODO: Write documentation.
    """
    def wrap(func: Callable[..., None]) -> Callable:
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            not_a_key = "Not a key"

            nonlocal press_key_to_start, exit_key

            input_queue: queue.Queue = queue.Queue()
            stop_event = threading.Event()
            input_thread = threading.Thread(
                target=input_deamon,
                args=(input_queue, stop_event, exit_key),
                daemon=True)
            input_thread.start()

            print("Entering keyboard interactive mode.")
            args[0].render()
            if press_key_to_start:
                print("Press a key to start...")
            key: Optional[str] = not_a_key
            is_started = not press_key_to_start
            while True:
                if not input_queue.empty():
                    key = input_queue.get()
                    if not is_started:
                        print("Go!")
                        key = not_a_key
                        is_started = True
                if key == exit_key:
                    print("Exiting keyboard interactive mode.")
                    stop_event.set()
                    time.sleep(0.01)
                    break
                if is_started:
                    try:
                        if key == not_a_key:
                            key = None
                        stop = func(*args, **kwargs, key=key)
                        key = not_a_key
                        if stop:
                            raise KeyboardInterrupt()
                    except KeyboardInterrupt:
                        key = exit_key
                    except Exception as e:  # pylint: disable=broad-except
                        print(e)
                        key = exit_key
                else:
                    time.sleep(0.1)
        return wrapped_func
    return wrap
