import os
import sys
import importlib


if importlib.util.find_spec("IPython") is not None:
    def interactive_mode() -> None:
        """Determine what kind of process is running Python kernel.

        :returns:
                - 0: Non-interactive process. It can be either Ipython console
                     or builtin terminaL. Does not support HTML embedding.
                - 1: Interactive Jupyter Notebook. (It can be confused with
                     Qtconsole)
                - 2: Interactive Google Colab. Not properly supported so far.
        """
        # Check if interactive display mode is disable
        if os.getenv("JIMINY_VIEWER_INTERACTIVE_DISABLE") == 1:
            return 0

        from IPython import get_ipython
        shell = get_ipython().__class__.__module__
        if shell == 'ipykernel.zmqshell':
            if 'spyder_kernels' in sys.modules:
                # Spyder is using Jupyter notebook as backend but is not able
                # to display HTML code in the IDE. So switching to
                # non-interactive mode.
                return 0
            # Jupyter notebook or qtconsole. Impossible to discriminate easily
            # without costly psutil inspection of the running process. So let's
            # assume it is Jupyter notebook, since nobody actually uses the
            # qtconsole anyway.
            return 1
        elif shell == 'IPython.terminal.interactiveshell':
            # Terminal running IPython
            return 0
        elif shell.startswith('google.colab.'):
            # Google Colaboratory
            return 2
        elif shell == 'builtins':
            # Terminal running Python
            return 0
        else:
            raise RuntimeError(f"Unknown Python environment: {shell}")
else:
    def interactive_mode() -> None:
        # Always return 0 if IPython module is not available
        return 0
