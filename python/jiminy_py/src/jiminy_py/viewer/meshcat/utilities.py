import os
import sys
import logging
from importlib.util import find_spec


if not os.getenv("JIMINY_INTERACTIVE_DISABLE", False) and \
        find_spec("IPython") is not None:
    # Get available ipykernel version (provided with jupyter)
    try:
        import ipykernel
        ipykernel_version_major = int(ipykernel.__version__[0])
        if ipykernel_version_major < 5:
            logging.warning(
                "Old ipykernel version < 5 not supported by interactive "
                "viewer. Update to avoid such limitation.")
    except ImportError:
        ipykernel_version_major = 0

    # Get shell class name
    from IPython import get_ipython
    shell = get_ipython().__class__.__module__

    def interactive_mode() -> int:
        """Determine what kind of process is running Python kernel.

        :returns:
            - 0: builtin terminal or plain script
            - 1: Spyder or Ipython console that does not support HTML embedding
            - 2: Interactive Jupyter Notebook with deprecated ipykernel
            - 3: Interactive Jupyter Notebook (can be confused with Qtconsole)
            - 4: Interactive Google Colab
        """
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
            if ipykernel_version_major < 5:
                return 2
            return 3
        elif shell == 'IPython.terminal.interactiveshell':
            # Terminal running IPython
            return 1
        elif shell.startswith('google.colab.'):
            # Google Colaboratory
            if ipykernel_version_major < 5:
                return 2
            return 4
        elif shell == 'builtins':
            # Terminal running Python
            return 0
        else:
            raise RuntimeError(f"Unknown Python environment: {shell}")

else:
    def interactive_mode() -> int:
        """Interactive mode forcibly disabled.
        """
        return 0
