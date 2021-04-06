
""" TODO: Write documentation.
"""
from typing import Optional, List, Tuple, Any

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QCursor, QPainter, QImage

from .panda3d_visualizer import Panda3dApp


class Panda3dQWidget(QWidget):
    """An interactive panda3D QWidget.
    """
    def __init__(self,
                 parent: Optional[Any] = None,
                 FPS: int = 60) -> None:
        """ TODO: Write documentation.
        """
        # Call base constructor
        super().__init__(parent)

        # Instantiate Panda3D app
        self._app = Panda3dApp(window_type='offscreen')

        # Enable mouse control
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._app.getMousePos = self.getMousePos
        self._app.taskMgr.add(self._app.moveOrbitalCameraTask,
                              "moveOrbitalCameraTask",
                              sort=2)

        # Create painter to render "screenshot" from panda3d
        self.paint_surface = QPainter()

        # Start event loop
        self.clock = QTimer()
        self.clock.setInterval(1000.0 / FPS)
        self.clock.timeout.connect(self.update)
        self.clock.start()

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute getter.

        It enables to get access to the attribute and methods of the low-level
        Panda3d app directly, without having to do it through `_app`.

        .. note::
            This method is not meant to be called manually.
        """
        return getattr(super().__getattribute__('_app'), name)

    def __dir__(self) -> List[str]:
        """Attribute lookup.

        It is mainly used by autocomplete feature of Ipython. It is overloaded
        to get consistent autocompletion wrt `getattr`.
        """
        return super().__dir__() + self._app.__dir__()

    def paintEvent(self, event: Any) -> None:
        """Pull the contents of the panda texture to the widget.
        """
        # Render the current scene
        self._app.step()

        # Get raw image and convert it to Qt format
        img = QImage(self._app.get_screenshot('BGRA', raw=True),
                     *self._app.getSize(),
                     QImage.Format_ARGB32).mirrored()

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
        pos = self.mapFromGlobal(QCursor().pos())
        return pos.x(), pos.y()

    def mousePressEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self._app.handleKey("mouse1", event.buttons() & Qt.LeftButton)
        self._app.handleKey("mouse2", event.buttons() & Qt.MiddleButton)
        self._app.handleKey("mouse3", event.buttons() & Qt.RightButton)

    def mouseReleaseEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self._app.handleKey("mouse1", event.buttons() & Qt.LeftButton)
        self._app.handleKey("mouse2", event.buttons() & Qt.MiddleButton)
        self._app.handleKey("mouse3", event.buttons() & Qt.RightButton)

    def wheelEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        delta = event.angleDelta().y()
        if delta > 0.0:
            self._app.handleKey("wheelup", True)
        else:
            self._app.handleKey("wheeldown", True)
