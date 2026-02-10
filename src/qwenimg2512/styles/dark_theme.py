"""Dark theme QSS stylesheet for the application."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

COLORS = {
    "bg_primary": "#1a1a2e",
    "bg_secondary": "#16213e",
    "bg_tertiary": "#0f3460",
    "bg_input": "#0d1b3e",
    "accent_primary": "#e94560",
    "accent_hover": "#ff6b81",
    "accent_success": "#4ade80",
    "accent_warning": "#fbbf24",
    "text_primary": "#ffffff",
    "text_secondary": "#a0a0b0",
    "text_muted": "#6b7280",
    "border": "#2a2a4a",
    "border_focus": "#e94560",
    "scrollbar_bg": "#1a1a2e",
    "scrollbar_handle": "#3a3a5e",
}


def get_stylesheet() -> str:
    c = COLORS
    return f"""
    QWidget {{
        background-color: {c["bg_primary"]};
        color: {c["text_primary"]};
        font-family: "Segoe UI", "Noto Sans", sans-serif;
        font-size: 13px;
    }}

    QMainWindow {{
        background-color: {c["bg_primary"]};
    }}

    QGroupBox {{
        background-color: {c["bg_secondary"]};
        border: 1px solid {c["border"]};
        border-radius: 8px;
        margin-top: 16px;
        padding: 16px 12px 12px 12px;
        font-weight: bold;
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 12px;
        color: {c["text_secondary"]};
    }}

    QLabel {{
        background: transparent;
        color: {c["text_primary"]};
    }}

    QLabel[class="muted"] {{
        color: {c["text_muted"]};
        font-size: 11px;
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {c["bg_input"]};
        border: 1px solid {c["border"]};
        border-radius: 6px;
        padding: 6px 10px;
        color: {c["text_primary"]};
        min-height: 28px;
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {c["border_focus"]};
    }}

    QPlainTextEdit {{
        background-color: {c["bg_input"]};
        border: 1px solid {c["border"]};
        border-radius: 6px;
        padding: 8px;
        color: {c["text_primary"]};
    }}

    QPlainTextEdit:focus {{
        border: 1px solid {c["border_focus"]};
    }}

    QPushButton {{
        background-color: {c["bg_tertiary"]};
        border: 1px solid {c["border"]};
        border-radius: 6px;
        padding: 8px 20px;
        color: {c["text_primary"]};
        font-weight: bold;
        min-height: 32px;
    }}

    QPushButton:hover {{
        background-color: {c["accent_primary"]};
        border-color: {c["accent_primary"]};
    }}

    QPushButton:pressed {{
        background-color: {c["accent_hover"]};
    }}

    QPushButton[class="primary"] {{
        background-color: {c["accent_primary"]};
        border-color: {c["accent_primary"]};
    }}

    QPushButton[class="primary"]:hover {{
        background-color: {c["accent_hover"]};
    }}

    QPushButton[class="danger"] {{
        background-color: #dc2626;
        border-color: #dc2626;
    }}

    QPushButton[class="danger"]:hover {{
        background-color: #ef4444;
    }}

    QComboBox {{
        background-color: {c["bg_input"]};
        border: 1px solid {c["border"]};
        border-radius: 6px;
        padding: 6px 10px;
        color: {c["text_primary"]};
        min-height: 28px;
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {c["bg_secondary"]};
        border: 1px solid {c["border"]};
        color: {c["text_primary"]};
        selection-background-color: {c["accent_primary"]};
    }}

    QCheckBox {{
        background: transparent;
        spacing: 8px;
    }}

    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {c["border"]};
        border-radius: 4px;
        background-color: {c["bg_input"]};
    }}

    QCheckBox::indicator:checked {{
        background-color: {c["accent_primary"]};
        border-color: {c["accent_primary"]};
    }}

    QProgressBar {{
        background-color: {c["bg_input"]};
        border: 1px solid {c["border"]};
        border-radius: 6px;
        text-align: center;
        color: {c["text_primary"]};
        min-height: 24px;
    }}

    QProgressBar::chunk {{
        background-color: {c["accent_primary"]};
        border-radius: 5px;
    }}

    QSlider::groove:horizontal {{
        background-color: {c["bg_input"]};
        border: 1px solid {c["border"]};
        height: 8px;
        border-radius: 4px;
    }}

    QSlider::handle:horizontal {{
        background-color: {c["accent_primary"]};
        border: none;
        width: 16px;
        height: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }}

    QScrollBar:vertical {{
        background: {c["scrollbar_bg"]};
        width: 10px;
        border: none;
    }}

    QScrollBar::handle:vertical {{
        background: {c["scrollbar_handle"]};
        min-height: 30px;
        border-radius: 5px;
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    QScrollBar:horizontal {{
        background: {c["scrollbar_bg"]};
        height: 10px;
        border: none;
    }}

    QScrollBar::handle:horizontal {{
        background: {c["scrollbar_handle"]};
        min-width: 30px;
        border-radius: 5px;
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    QMenuBar {{
        background-color: {c["bg_secondary"]};
        border-bottom: 1px solid {c["border"]};
    }}

    QMenuBar::item:selected {{
        background-color: {c["accent_primary"]};
    }}

    QMenu {{
        background-color: {c["bg_secondary"]};
        border: 1px solid {c["border"]};
    }}

    QMenu::item:selected {{
        background-color: {c["accent_primary"]};
    }}

    QStatusBar {{
        background-color: {c["bg_secondary"]};
        border-top: 1px solid {c["border"]};
        color: {c["text_secondary"]};
    }}

    QSplitter::handle {{
        background-color: {c["border"]};
        height: 2px;
    }}
    """


def apply_dark_theme(app: QApplication) -> None:
    app.setStyleSheet(get_stylesheet())
