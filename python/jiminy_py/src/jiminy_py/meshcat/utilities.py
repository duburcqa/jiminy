import importlib


if importlib.util.find_spec("IPython") is not None:
    def is_notebook() -> None:
        """Determine whether Python is running inside a Notebook or not.
        """
        from IPython import get_ipython
        shell = get_ipython().__class__.__module__
        if shell == 'ipykernel.zmqshell':
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
    def is_notebook() -> None:
        # Always return 0 if ipython is not available
        return 0
