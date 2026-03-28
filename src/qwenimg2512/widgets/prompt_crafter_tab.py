"""Prompt Crafter tab — recipe-based prompt builder for Qwen Image Edit."""

from __future__ import annotations

import logging
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from qwenimg2512.prompt_recipes import (
    BUILTIN_RECIPES,
    PromptRecipe,
    load_custom_recipes,
    save_custom_recipes,
)

logger = logging.getLogger(__name__)


class PromptCrafterTabWidget(QWidget):
    """Interactive prompt builder with LoRA-aware recipe templates."""

    # (prompt, lora_repo, lora_weights)
    send_to_edit = Signal(str, str, str)
    send_to_edit_2509 = Signal(str, str, str)
    send_to_generate = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._custom_recipes: list[PromptRecipe] = load_custom_recipes()
        self._all_recipes: list[PromptRecipe] = list(BUILTIN_RECIPES) + self._custom_recipes
        self._placeholder_edits: dict[str, QLineEdit] = {}
        self._selected_recipe: PromptRecipe | None = None
        self._setup_ui()
        self._refresh_recipe_list()
        # Select first item if available
        if self.recipe_list.count() > 0:
            self.recipe_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    #  UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(8)

        # ── 1. Category filter & recipe list ──────────────────────────
        browser_group = QGroupBox("Recipe Browser")
        browser_layout = QVBoxLayout(browser_group)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        categories = sorted({r.category for r in self._all_recipes})
        for cat in categories:
            self.category_combo.addItem(cat)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        filter_row.addWidget(self.category_combo, 1)
        browser_layout.addLayout(filter_row)

        self.recipe_list = QListWidget()
        self.recipe_list.setMaximumHeight(200)
        self.recipe_list.currentRowChanged.connect(self._on_recipe_selected)
        browser_layout.addWidget(self.recipe_list)

        # Tip label
        self.tip_label = QLabel()
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet(
            "color: #8888aa; font-style: italic; padding: 2px 4px;"
        )
        browser_layout.addWidget(self.tip_label)

        content_layout.addWidget(browser_group)

        # ── 2. Placeholder form ──────────────────────────────────────
        self.placeholder_group = QGroupBox("Template Fields")
        self.placeholder_layout = QVBoxLayout(self.placeholder_group)
        self.placeholder_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.addWidget(self.placeholder_group)

        # ── 3. Live preview ───────────────────────────────────────────
        preview_group = QGroupBox("Assembled Prompt (preview)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setMaximumHeight(120)
        self.preview_edit.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; font-family: monospace; "
            "border: 1px solid #333355; border-radius: 4px; padding: 4px;"
        )
        preview_layout.addWidget(self.preview_edit)
        content_layout.addWidget(preview_group)

        # ── 4. LoRA info ──────────────────────────────────────────────
        lora_group = QGroupBox("Linked LoRA")
        lora_layout = QVBoxLayout(lora_group)
        self.lora_info_label = QLabel("No LoRA")
        self.lora_info_label.setWordWrap(True)
        self.lora_info_label.setStyleSheet("color: #aaaacc; font-size: 12px;")
        self.lora_info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lora_layout.addWidget(self.lora_info_label)

        images_row = QHBoxLayout()
        images_row.addWidget(QLabel("Ref images needed:"))
        self.num_images_label = QLabel("1")
        self.num_images_label.setStyleSheet("font-weight: bold;")
        images_row.addWidget(self.num_images_label)
        images_row.addStretch()
        lora_layout.addLayout(images_row)

        content_layout.addWidget(lora_group)

        # ── 5. Send-to buttons ────────────────────────────────────────
        send_group = QGroupBox("Send Crafted Prompt To…")
        send_layout = QHBoxLayout(send_group)

        self.btn_edit = QPushButton("→ Edit 2511")
        self.btn_edit.setToolTip(
            "Copy the assembled prompt (and LoRA) to the Edit 2511 tab and switch to it."
        )
        self.btn_edit.clicked.connect(self._send_to_edit)
        send_layout.addWidget(self.btn_edit)

        self.btn_edit_2509 = QPushButton("→ Edit 2509")
        self.btn_edit_2509.setToolTip(
            "Copy the assembled prompt (and LoRA) to the Edit 2509 tab and switch to it."
        )
        self.btn_edit_2509.clicked.connect(self._send_to_edit_2509)
        send_layout.addWidget(self.btn_edit_2509)

        self.btn_generate = QPushButton("→ Generate")
        self.btn_generate.setToolTip(
            "Copy the assembled prompt to the Generate tab and switch to it."
        )
        self.btn_generate.clicked.connect(self._send_to_generate)
        send_layout.addWidget(self.btn_generate)

        content_layout.addWidget(send_group)

        # ── 6. Recipe management ──────────────────────────────────────
        manage_group = QGroupBox("Custom Recipes")
        manage_layout = QHBoxLayout(manage_group)

        self.btn_save_custom = QPushButton("Save Current as Custom")
        self.btn_save_custom.setToolTip(
            "Save the current template + placeholder defaults as a new custom recipe."
        )
        self.btn_save_custom.clicked.connect(self._save_as_custom)
        manage_layout.addWidget(self.btn_save_custom)

        self.btn_delete_custom = QPushButton("Delete Selected Custom")
        self.btn_delete_custom.setToolTip("Delete the currently selected custom recipe.")
        self.btn_delete_custom.clicked.connect(self._delete_custom)
        self.btn_delete_custom.setEnabled(False)
        manage_layout.addWidget(self.btn_delete_custom)

        content_layout.addWidget(manage_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    # ------------------------------------------------------------------
    #  Recipe list helpers
    # ------------------------------------------------------------------

    def _refresh_recipe_list(self) -> None:
        """Rebuild the recipe list widget from the current filter."""
        self._all_recipes = list(BUILTIN_RECIPES) + self._custom_recipes
        # Refresh category combo while preserving selection
        current_cat = self.category_combo.currentText()
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItem("All")
        for cat in sorted({r.category for r in self._all_recipes}):
            self.category_combo.addItem(cat)
        idx = self.category_combo.findText(current_cat)
        self.category_combo.setCurrentIndex(max(idx, 0))
        self.category_combo.blockSignals(False)

        selected_cat = self.category_combo.currentText()
        self.recipe_list.blockSignals(True)
        self.recipe_list.clear()
        for recipe in self._all_recipes:
            if selected_cat != "All" and recipe.category != selected_cat:
                continue
            prefix = "⭐ " if not recipe.builtin else ""
            item = QListWidgetItem(f"{prefix}{recipe.name}")
            item.setToolTip(recipe.tip or recipe.template)
            item.setData(Qt.ItemDataRole.UserRole, recipe)
            self.recipe_list.addItem(item)
        self.recipe_list.blockSignals(False)

    def _on_category_changed(self, _text: str) -> None:
        self._refresh_recipe_list()
        if self.recipe_list.count() > 0:
            self.recipe_list.setCurrentRow(0)
        else:
            self._clear_selection()

    # ------------------------------------------------------------------
    #  Recipe selection
    # ------------------------------------------------------------------

    def _on_recipe_selected(self, row: int) -> None:
        if row < 0:
            self._clear_selection()
            return
        item = self.recipe_list.item(row)
        if item is None:
            self._clear_selection()
            return
        recipe: PromptRecipe = item.data(Qt.ItemDataRole.UserRole)
        self._selected_recipe = recipe
        self._populate_placeholder_form(recipe)
        self._update_lora_info(recipe)
        self._rebuild_preview()

        # Enable delete only for custom recipes
        self.btn_delete_custom.setEnabled(not recipe.builtin)

    def _clear_selection(self) -> None:
        self._selected_recipe = None
        self._clear_placeholder_form()
        self.preview_edit.setPlainText("")
        self.lora_info_label.setText("No LoRA")
        self.num_images_label.setText("–")
        self.tip_label.setText("")
        self.btn_delete_custom.setEnabled(False)

    # ------------------------------------------------------------------
    #  Placeholder form
    # ------------------------------------------------------------------

    def _clear_placeholder_form(self) -> None:
        """Remove all dynamic placeholder widgets."""
        while self.placeholder_layout.count():
            child = self.placeholder_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                # remove sub-layout children
                while child.layout().count():
                    sub = child.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        self._placeholder_edits.clear()

    def _populate_placeholder_form(self, recipe: PromptRecipe) -> None:
        """Create QLineEdits for each placeholder in the template."""
        self._clear_placeholder_form()

        if not recipe.placeholders:
            lbl = QLabel("No editable fields — the template is used as-is.")
            lbl.setStyleSheet("color: #888; font-style: italic;")
            self.placeholder_layout.addWidget(lbl)
            return

        for ph in recipe.placeholders:
            row = QHBoxLayout()
            label = QLabel(ph.replace("_", " ").title() + ":")
            label.setMinimumWidth(100)
            row.addWidget(label)

            edit = QLineEdit()
            edit.setPlaceholderText(recipe.placeholder_defaults.get(ph, ""))
            edit.setText(recipe.placeholder_defaults.get(ph, ""))
            edit.textChanged.connect(self._rebuild_preview)
            row.addWidget(edit, 1)

            self.placeholder_layout.addLayout(row)
            self._placeholder_edits[ph] = edit

        self.tip_label.setText(recipe.tip)

    # ------------------------------------------------------------------
    #  Live preview
    # ------------------------------------------------------------------

    def _rebuild_preview(self) -> None:
        """Assemble the prompt from template + current field values."""
        recipe = self._selected_recipe
        if recipe is None:
            self.preview_edit.setPlainText("")
            return

        values: dict[str, str] = {}
        for ph in recipe.placeholders:
            edit = self._placeholder_edits.get(ph)
            if edit is not None:
                values[ph] = edit.text()
            else:
                values[ph] = recipe.placeholder_defaults.get(ph, "")

        try:
            prompt = recipe.template.format(**values)
        except (KeyError, IndexError):
            prompt = recipe.template

        self.preview_edit.setPlainText(prompt.strip())

    # ------------------------------------------------------------------
    #  LoRA info
    # ------------------------------------------------------------------

    def _update_lora_info(self, recipe: PromptRecipe) -> None:
        if recipe.lora_repo:
            self.lora_info_label.setText(
                f"<b>Repo:</b> {recipe.lora_repo}<br>"
                f"<b>Weights:</b> {recipe.lora_weights}"
            )
        else:
            self.lora_info_label.setText("No LoRA linked (freeform recipe)")
        self.num_images_label.setText(str(recipe.num_images))

    # ------------------------------------------------------------------
    #  Send-to actions
    # ------------------------------------------------------------------

    def _get_assembled_prompt(self) -> str:
        return self.preview_edit.toPlainText().strip()

    def _send_to_edit(self) -> None:
        prompt = self._get_assembled_prompt()
        if not prompt:
            return
        recipe = self._selected_recipe
        repo = recipe.lora_repo if recipe else ""
        weights = recipe.lora_weights if recipe else ""
        self.send_to_edit.emit(prompt, repo, weights)

    def _send_to_edit_2509(self) -> None:
        prompt = self._get_assembled_prompt()
        if not prompt:
            return
        recipe = self._selected_recipe
        repo = recipe.lora_repo if recipe else ""
        weights = recipe.lora_weights if recipe else ""
        self.send_to_edit_2509.emit(prompt, repo, weights)

    def _send_to_generate(self) -> None:
        prompt = self._get_assembled_prompt()
        if prompt:
            self.send_to_generate.emit(prompt)

    # ------------------------------------------------------------------
    #  Custom recipe management
    # ------------------------------------------------------------------

    def _save_as_custom(self) -> None:
        """Save the current state as a new custom recipe."""
        name, ok = QInputDialog.getText(
            self, "Save Custom Recipe", "Recipe name:"
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Determine category
        cat, ok2 = QInputDialog.getText(
            self,
            "Recipe Category",
            "Category (e.g. Style Transfer, Lighting, Custom):",
            text="Custom",
        )
        if not ok2:
            cat = "Custom"

        # Build template from current state
        template = self._get_assembled_prompt()
        if not template:
            QMessageBox.warning(self, "Empty Prompt", "The assembled prompt is empty.")
            return

        new_recipe = PromptRecipe(
            name=name,
            category=cat.strip() or "Custom",
            template=template,
            placeholders=[],
            placeholder_defaults={},
            lora_repo="",
            lora_weights="",
            tip="User-created recipe.",
            num_images=1,
            builtin=False,
        )
        self._custom_recipes.append(new_recipe)
        save_custom_recipes(self._custom_recipes)
        self._refresh_recipe_list()

        # Select the newly created recipe
        for i in range(self.recipe_list.count()):
            item = self.recipe_list.item(i)
            r: PromptRecipe = item.data(Qt.ItemDataRole.UserRole)
            if r.name == name and not r.builtin:
                self.recipe_list.setCurrentRow(i)
                break

        QMessageBox.information(
            self, "Saved", f"Custom recipe '{name}' saved successfully."
        )

    def _delete_custom(self) -> None:
        """Delete the selected custom recipe."""
        recipe = self._selected_recipe
        if recipe is None or recipe.builtin:
            return
        reply = QMessageBox.question(
            self,
            "Delete Recipe",
            f"Delete custom recipe '{recipe.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._custom_recipes = [r for r in self._custom_recipes if r is not recipe]
        save_custom_recipes(self._custom_recipes)
        self._refresh_recipe_list()
        if self.recipe_list.count() > 0:
            self.recipe_list.setCurrentRow(0)
        else:
            self._clear_selection()
