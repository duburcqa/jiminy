""" TODO: Write documentation.
"""
from typing import Optional, Tuple, Any

from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from .panda3d_visualizer import Panda3dViewer


FRAMERATE = 30


Qt = QtCore.Qt


class Panda3dQWidget(Panda3dViewer, QtWidgets.QWidget):
    """An interactive panda3D QWidget.
    """
    def __init__(self, parent: Optional[Any] = None) -> None:
        """ TODO: Write documentation.
        """
        # Initialize Qt widget
        QtWidgets.QWidget.__init__(self, parent=parent)

        # Initialize Panda3D app
        Panda3dViewer.__init__(self, window_type='offscreen')

        # Only accept focus by clicking on widget
        self.setFocusPolicy(Qt.ClickFocus)

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

    def destroy(self):
        super(Panda3dViewer, self).destroy()
        super(QtWidgets.QWidget, self).destroy()

    def close(self) -> bool:
        super(Panda3dViewer, self).destroy()
        return super(QtWidgets.QWidget, self).close()

    def paintEvent(self, event: Any) -> None:
        """Pull the contents of the panda texture to the widget.
        """
        # Get raw image and convert it to Qt format.
        # Note that `QImage` apparently does not manage the lifetime of the
        # input data buffer, so it is necessary to keep it is local scope.
        data = self.get_screenshot('RGBA', raw=True)
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
        self.set_window_size(
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
