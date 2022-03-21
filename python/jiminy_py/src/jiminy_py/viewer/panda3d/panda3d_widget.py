""" TODO: Write documentation.
"""
from typing import Optional, Tuple, Any

from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui

from .panda3d_visualizer import Panda3dApp


FRAMERATE = 30


class Panda3dQWidget(Panda3dApp, QtWidgets.QWidget):
    """An interactive panda3D QWidget.
    """
    def __init__(self, parent: Optional[Any] = None) -> None:
        """ TODO: Write documentation.
        """
        # Initialize Qt widget
        QtWidgets.QWidget.__init__(self, parent=parent)

        # Initialize Panda3D app
        Panda3dApp.__init__(self)

        # Only accept focus by clicking on widget
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

        # Configure mouse control
        self.setMouseTracking(True)

        # Create painter to render "screenshot" from panda3d
        self.paint_surface = QtGui.QPainter()

        # Start event loop
        self.clock = QtCore.QTimer()
        self.clock.setInterval(1000.0 / FRAMERATE)
        self.clock.timeout.connect(self.update)
        self.clock.start()

    def destroy(self) -> None:
        Panda3dApp.destroy(self)
        QtWidgets.QWidget.destroy(self)

    def close(self) -> bool:
        Panda3dApp.destroy(self)
        return QtWidgets.QWidget.close(self)

    def paintEvent(self, event: Any) -> None:
        """Pull the contents of the panda texture to the widget.
        """
        # Updating the pose of the camera
        self.move_orbital_camera_task()

        # Get raw image and convert it to Qt format.
        # Note that `QImage` does not manage the lifetime of the input data
        # buffer, so it is necessary to keep it is local scope until the end of
        # its drawning.
        data = self.get_screenshot('RGB', raw=True)
        img = QtGui.QImage(
            data, *self.buff.getSize(), QtGui.QImage.Format_RGB888)

        # Render image on Qt widget
        self.paint_surface.begin(self)
        self.paint_surface.drawImage(0, 0, img)
        self.paint_surface.end()

    def resizeEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self.set_window_size(
            event.size().width(), event.size().height())

    def getMousePos(self) -> Tuple[float, float]:
        """ TODO: Write documentation.
        """
        pos = self.mapFromGlobal(QtGui.QCursor().pos())
        return pos.x(), pos.y()

    def mousePressEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self.handle_key("mouse1", event.buttons() & QtCore.Qt.LeftButton)
        self.handle_key("mouse2", event.buttons() & QtCore.Qt.MiddleButton)
        self.handle_key("mouse3", event.buttons() & QtCore.Qt.RightButton)

    def mouseReleaseEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        self.handle_key("mouse1", event.buttons() & QtCore.Qt.LeftButton)
        self.handle_key("mouse2", event.buttons() & QtCore.Qt.MiddleButton)
        self.handle_key("mouse3", event.buttons() & QtCore.Qt.RightButton)

    def wheelEvent(self, event: Any) -> None:
        """ TODO: Write documentation.
        """
        delta = event.angleDelta().y()
        if delta > 0.0:
            self.handle_key("wheelup", True)
        else:
            self.handle_key("wheeldown", True)
