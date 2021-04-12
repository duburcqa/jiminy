""" TODO: Write documentation.
"""
from typing import Optional, List, Tuple, Any

from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from .panda3d_visualizer import Panda3dApp


FRAMERATE = 30


Qt = QtCore.Qt


class Panda3dQWidget(QtWidgets.QWidget):
    """An interactive panda3D QWidget.
    """
    def __init__(self, parent: Optional[Any] = None) -> None:
        """ TODO: Write documentation.
        """
        # Call base constructor
        super().__init__(parent)

        # Only accept focus by clicking on widget
        self.setFocusPolicy(Qt.ClickFocus)

        # Instantiate Panda3D app
        self._app = Panda3dApp(window_type='offscreen')

        # Enable mouse control
        self.setMouseTracking(True)
        self._app.getMousePos = self.getMousePos
        self._app.taskMgr.add(
            self._app.move_orbital_camera_task, "move_camera_task", sort=2)

        # Create painter to render "screenshot" from panda3d
        self.paint_surface = QtGui.QPainter()

        # Start event loop
        self.clock = QtCore.QTimer()
        self.clock.setInterval(1000.0 / FRAMERATE)
        self.clock.timeout.connect(self.update)
        self.clock.start()

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Panda3d app directly, without having to do it through `_app`.

        .. note::
            This method is not meant to be called manually.
        """
        return getattr(self.__getattribute__('_app'), name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self._app.__dir__()

    def close(self) -> bool:
        self._app.destroy()
        return super().close()

    def paintEvent(self, event: Any) -> None:
        """Pull the contents of the panda texture to the widget.
        """
        # Render the scene
        self._app.step()

        # Get raw image and convert it to Qt format.
        # Note that `QImage` apparently does not manage the lifetime of the
        # input data buffer, so it is necessary to keep it is local scope.
        data = self._app.get_screenshot('RGBA', raw=True)
        img = QtGui.QImage(data,
                           *self._app.buff.getSize(),
                           QtGui.QImage.Format_RGBA8888).mirrored()

        # Render image on Qt widget
        self.paint_surface.begin(self)
        self.paint_surface.drawImage(0, 0, img)
        self.paint_surface.end()

    def resizeEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self._app.set_window_size(
            event.size().width(), event.size().height())

    def getMousePos(self) -> Tuple[int, int]:
        """ TODO: Write documentation.
        """
        pos = self.mapFromGlobal(QtGui.QCursor().pos())
        return pos.x(), pos.y()

    def mousePressEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self._app.handle_key("mouse1", event.buttons() & Qt.LeftButton)
        self._app.handle_key("mouse2", event.buttons() & Qt.MiddleButton)
        self._app.handle_key("mouse3", event.buttons() & Qt.RightButton)

    def mouseReleaseEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self._app.handle_key("mouse1", event.buttons() & Qt.LeftButton)
        self._app.handle_key("mouse2", event.buttons() & Qt.MiddleButton)
        self._app.handle_key("mouse3", event.buttons() & Qt.RightButton)

    def wheelEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        delta = event.angleDelta().y()
        if delta > 0.0:
            self._app.handle_key("wheelup", True)
        else:
            self._app.handle_key("wheeldown", True)
