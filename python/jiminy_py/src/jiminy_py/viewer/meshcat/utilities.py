import os
import sys
from importlib.util import find_spec


if find_spec("IPython") is not None:
    def interactive_mode() -> int:
        """Determine what kind of process is running Python kernel.

        :returns:
            - 0: builtin terminal
            - 1: Spyder or Ipython console. Does not support HTML embedding
            - 2: Interactive Jupyter Notebook (can be confused with Qtconsole)
            - 3: Interactive Google Colab
        """
        # Check if interactive display mode is disable
        if os.getenv("JIMINY_INTERACTIVE_DISABLE") == 1:
            return 0

        from IPython import get_ipython
        shell = get_ipython().__class__.__module__
        if shell == 'ipykernel.zmqshell':
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
        elif shell == 'IPython.terminal.interactiveshell':
            # Terminal running IPython
            return 1
        elif shell.startswith('google.colab.'):
            # Google Colaboratory
            return 3
        elif shell == 'builtins':
            # Terminal running Python
            return 0
        else:
            raise RuntimeError(f"Unknown Python environment: {shell}")
else:
    def interactive_mode() -> int:
        # Always return 0 if IPython module is not available
        return 0
