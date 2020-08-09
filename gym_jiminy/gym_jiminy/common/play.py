import os
import threading
import queue
import time


class Getch():
    """
    @brief   Gets a single character from standard input.

    @details Does not echo to the screen.
    """

    def __init__(self, stop_event = None):
        self.stop_event = stop_event
        if os.name != 'nt':
            import sys, fcntl, termios
            self.fd = sys.stdin.fileno()
            self.oldterm = termios.tcgetattr(self.fd)
            newattr = termios.tcgetattr(self.fd)
            newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSANOW, newattr)
            self.oldflags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            newflags = self.oldflags | os.O_NONBLOCK
            fcntl.fcntl(self.fd, fcntl.F_SETFL, newflags)

    def __del__(self):
        if os.name != 'nt':
            import fcntl, termios, tty
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.oldterm)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.oldflags)

    def __call__(self):
        if os.name != 'nt':
            c = ''
            try:
                import sys
                import termios
                termios.tcflush(self.fd, termios.TCIFLUSH)
                while self.stop_event is None or \
                        not self.stop_event.is_set():
                    try:
                        c += sys.stdin.read(1)
                        if c and (c[:1] != '\x1b' or len(c) > 2):
                            break
                    except IOError:
                        pass
            finally:
                return c
        else:
            import msvcrt
            while self.stop_event is None or \
                    not self.stop_event.is_set():
                if msvcrt.kbhit():
                    return msvcrt.getch()

def input_deamon(input_queue, stop_event, exit_key):
    CHAR_TO_ARROW_MAPPING = {"\x1b[A" : "Up",
                             "\x1b[B": "Down",
                             "\x1b[C": "Right",
                             "\x1b[D": "Left"}
    getch = Getch(stop_event)
    while not stop_event.is_set():
        c = getch()
        if c in CHAR_TO_ARROW_MAPPING.keys():
            c = CHAR_TO_ARROW_MAPPING[c]
        if list(bytes(c.encode('utf-8'))) == [3]:
            c = exit_key
        input_queue.put(c)
    del getch

def loop_interactive(press_key_to_start=True, exit_key='k'):
    def wrap(func):
        def wrapped_func(*args, **kwargs):
            NOT_A_KEY = "Not a key"

            nonlocal press_key_to_start, exit_key

            input_queue = queue.Queue()
            stop_event = threading.Event()
            input_thread = threading.Thread(target=input_deamon,
                                            args=(input_queue, stop_event, exit_key),
                                            daemon=True)
            input_thread.start()

            print("Starting keyboard interactive mode.")
            if press_key_to_start:
                print("Press a key to start...")
            key = NOT_A_KEY
            is_started = not press_key_to_start
            while True:
                if (not input_queue.empty()):
                    key = input_queue.get()
                    if not is_started:
                        print("Go!")
                        key = NOT_A_KEY
                        is_started = False
                if key == exit_key:
                    print("Exiting keyboard interactive mode.")
                    stop_event.set()
                    time.sleep(0.01)
                    break
                if is_started:
                    try:
                        if key == NOT_A_KEY:
                            key = None
                        stop = func(*args, **kwargs, key=key)
                        key = NOT_A_KEY
                        if stop:
                            raise KeyboardInterrupt()
                    except KeyboardInterrupt:
                        key = exit_key
                    except Exception as err:
                        print(err)
                        key = exit_key
                else:
                    time.sleep(0.1)
        return wrapped_func
    return wrap
