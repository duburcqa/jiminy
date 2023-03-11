""" TODO: Write documentation.
"""
import os
import sys
from importlib.util import find_spec


if (find_spec("IPython") is not None and
        "JIMINY_INTERACTIVE_DISABLE" not in os.environ):
    # Get shell class name
    from IPython import get_ipython
    SHELL = get_ipython().__class__.__module__

    def interactive_mode() -> int:
        """Determine what kind of process is running Python kernel.

        :returns:
            - 0: builtin terminal or plain script
            - 1: Spyder or Ipython console that does not support HTML embedding
            - 2: Interactive Jupyter Notebook (can be confused with Qtconsole)
            - 3: Interactive Google Colab
        """
        if SHELL.startswith('ipykernel.zmqshell'):
            if 'spyder_kernels' in sys.modules:
                # Spyder is using Jupyter notebook as backend but is not able
                # to display HTML code in the IDE. So switching to
                # non-interactive mode.
                return 1
            # Jupyter notebook or qtconsole. Impossible to discriminate easily
            # without costly psutil inspection of the running process. So let's
            # assume it is Jupyter notebook, since nobody actually uses the
            # qtconsole anyway.
            return 2
        if SHELL.startswith('IPython.terminal'):
            # Terminal running IPython
            return 1
        if SHELL.startswith('google.colab'):
            return 3
        if SHELL == 'builtins':
            # Terminal running Python
            return 0
        raise RuntimeError(f"Unknown Python environment: {SHELL}")

else:
    def interactive_mode() -> int:
        """Interactive mode forcibly disabled.
        """
        return 0
