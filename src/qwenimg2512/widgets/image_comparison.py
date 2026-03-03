"""Interactive image comparison widget with a draggable slider."""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QWidget, QSizePolicy


class ImageComparisonWidget(QWidget):
    """A widget that displays two images side-by-side with a draggable boundary."""
    
    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        
        self._pixmap_base: QPixmap | None = None
        self._pixmap_overlay: QPixmap | None = None
        self._slider_pos: float = 0.5  # 0.0 to 1.0
        self._is_dragging: bool = False
        
        # We need to maintain the scaled pixmaps to draw them efficiently
        self._scaled_base: QPixmap | None = None
        self._scaled_overlay: QPixmap | None = None
        self._draw_rect: QRect = QRect()

    def set_images(self, pixmap_base: QPixmap, pixmap_overlay: QPixmap | None) -> None:
        """Set the base and overlay images. Overlay can be None for single image mode."""
        self._pixmap_base = pixmap_base
        self._pixmap_overlay = pixmap_overlay
        self._slider_pos = 0.5
        self._update_scaled_pixmaps()
        self.update()

    def clear(self) -> None:
        self._pixmap_base = None
        self._pixmap_overlay = None
        self._scaled_base = None
        self._scaled_overlay = None
        self._draw_rect = QRect()
        self.update()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmaps()

    def _update_scaled_pixmaps(self) -> None:
        if not self._pixmap_base or self._pixmap_base.isNull():
            self._scaled_base = None
            self._scaled_overlay = None
            return

        # Keep aspect ratio, scale to widget size
        size = self.size()
        self._scaled_base = self._pixmap_base.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        
        # Calculate where to draw so it's centered
        x = (size.width() - self._scaled_base.width()) // 2
        y = (size.height() - self._scaled_base.height()) // 2
        self._draw_rect = QRect(x, y, self._scaled_base.width(), self._scaled_base.height())

        if self._pixmap_overlay and not self._pixmap_overlay.isNull():
            self._scaled_overlay = self._pixmap_overlay.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            self._scaled_overlay = None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw border/placeholder if no image
        if not self._scaled_base:
            painter.setPen(QPen(QColor("#2a2a4a"), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
            painter.setPen(QPen(QColor("#cccccc")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image generated yet")
            return

        # Draw base image
        painter.drawPixmap(self._draw_rect.topLeft(), self._scaled_base)

        # Draw overlay image if it exists
        if self._scaled_overlay:
            # Calculate split pixel purely based on the draw rect
            split_x = self._draw_rect.left() + int(self._draw_rect.width() * self._slider_pos)
            
            # Clip rect for overlay (left side)
            clip_rect = QRect(
                self._draw_rect.left(),
                self._draw_rect.top(),
                split_x - self._draw_rect.left(),
                self._draw_rect.height()
            )
            
            painter.setClipRect(clip_rect)
            painter.drawPixmap(self._draw_rect.topLeft(), self._scaled_overlay)
            painter.setClipping(False)

            # Draw slider line
            pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(pen)
            painter.drawLine(split_x, self._draw_rect.top(), split_x, self._draw_rect.bottom())
            
            # Draw slider handle
            handle_y = self._draw_rect.center().y()
            handle_rect = QRect(split_x - 10, handle_y - 20, 20, 40)
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(handle_rect, 4, 4)
            
            # Draw arrows on handle
            painter.setPen(QPen(QColor(50, 50, 50), 2))
            painter.drawLine(split_x - 4, handle_y - 5, split_x - 8, handle_y)
            painter.drawLine(split_x - 4, handle_y + 5, split_x - 8, handle_y)
            
            painter.drawLine(split_x + 4, handle_y - 5, split_x + 8, handle_y)
            painter.drawLine(split_x + 4, handle_y + 5, split_x + 8, handle_y)

    def _get_slider_x(self) -> int:
        if not self._draw_rect.isValid():
            return 0
        return self._draw_rect.left() + int(self._draw_rect.width() * self._slider_pos)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._scaled_overlay:
            split_x = self._get_slider_x()
            # Allow grabbing within a reasonable margin of the line
            if abs(event.pos().x() - split_x) < 20 and self._draw_rect.contains(event.pos()):
                self._is_dragging = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._scaled_overlay:
            split_x = self._get_slider_x()
            # Update cursor shape based on hover
            if self._is_dragging:
                # Clamp position inside the image bounds
                new_x = max(self._draw_rect.left(), min(event.pos().x(), self._draw_rect.right()))
                self._slider_pos = (new_x - self._draw_rect.left()) / max(1, self._draw_rect.width())
                self.update()
                event.accept()
                return
            elif abs(event.pos().x() - split_x) < 20 and self._draw_rect.contains(event.pos()):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
