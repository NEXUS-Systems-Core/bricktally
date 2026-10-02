"""Operator window. Core counting lives in bricktally, not here."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from bricktally import __version__
from bricktally.apply_update import spawn_updater
from bricktally.brickognize import crop_around, identify_image
from bricktally.camera import open_camera
from bricktally.colors import Palette
from bricktally.config import UPDATE_REPO
from bricktally.count import CountResult, correct_piece, count_image, draw_overlay, rematch_unlocked
from bricktally.inventory import Inventory
from bricktally.bsx import to_bsx, validate_bsx
from bricktally.settings import Settings
from bricktally.updates import check_for_update, download_verified, repo_configured
from bricktally.wb import grey_card_gains


class _Job(QThread):
    ok = Signal(object)
    err = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.ok.emit(self._fn())
        except Exception as exc:
            self.err.emit(str(exc))


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"BrickTally {__version__}")
        self.resize(1180, 760)
        self.settings = Settings().load()
        self.palette = Palette.load()
        self.palette.set_samples(self.settings.user_samples())
        self.inventory = Inventory.from_json(self.settings.inventory)
        self.cap = None
        self.live = True
        self.frame: np.ndarray | None = None
        self.raw: np.ndarray | None = None
        self.still: np.ndarray | None = None
        self.count_result: CountResult | None = None
        self.selected_index: int | None = None
        self.part_id = ""
        self.part_name = ""
        self.pending_update = None
        self._job: _Job | None = None
        self._build()
        self._open_camera(self.settings.camera_index)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._preview)
        self.timer.start(70)
        QTimer.singleShot(400, self._check_updates)

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root.setStyleSheet(
            "QPushButton { min-height: 46px; font-size: 16px; padding: 6px 14px; }"
            "QLabel { font-size: 15px; }"
            "QTableWidget { font-size: 14px; }"
            "QComboBox, QLineEdit { min-height: 36px; font-size: 15px; }"
        )
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera"))
        self.camera_combo = QComboBox()
        for index in range(6):
            self.camera_combo.addItem(f"Camera {index + 1}", index)
        self.camera_combo.setCurrentIndex(min(self.settings.camera_index, 5))
        self.camera_combo.currentIndexChanged.connect(self._camera_changed)
        top.addWidget(self.camera_combo)
        top.addSpacing(16)
        top.addWidget(QLabel("Condition"))
        self.condition_combo = QComboBox()
        self.condition_combo.addItem("Used", "U")
        self.condition_combo.addItem("New", "N")
        self.condition_combo.setCurrentIndex(0 if self.settings.condition == "U" else 1)
        self.condition_combo.currentIndexChanged.connect(self._condition_changed)
        top.addWidget(self.condition_combo)
        top.addSpacing(16)
        top.addWidget(QLabel("Piece px"))
        self.piece_area = QSpinBox()
        self.piece_area.setRange(0, 200000)
        self.piece_area.setSingleStep(100)
        self.piece_area.setSpecialValueText("auto")
        self.piece_area.setValue(int(self.settings.piece_area or 0))
        self.piece_area.editingFinished.connect(self._piece_area_changed)
        top.addWidget(self.piece_area)
        top.addStretch(1)
        layout.addLayout(top)

        body = QHBoxLayout()
        self.photo = QLabel("Starting camera...")
        self.photo.setMinimumSize(640, 480)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo.setStyleSheet("background: #222; color: white;")
        self.photo.mousePressEvent = self._photo_clicked
        body.addWidget(self.photo, stretch=3)

        side = QVBoxLayout()
        self.part_label = QLabel("Part: not chosen yet")
        self.part_label.setWordWrap(True)
        side.addWidget(self.part_label)
        self.candidates = QVBoxLayout()
        side.addLayout(self.candidates)
        manual = QHBoxLayout()
        self.part_entry = QLineEdit()
        self.part_entry.setPlaceholderText("Type a part number")
        use_btn = QPushButton("Use this part")
        use_btn.clicked.connect(self._use_typed_part)
        manual.addWidget(self.part_entry)
        manual.addWidget(use_btn)
        side.addLayout(manual)

        side.addWidget(QLabel("This pile"))
        self.totals = QTableWidget(0, 2)
        self.totals.setHorizontalHeaderLabels(["Color", "Qty"])
        self.totals.horizontalHeader().setStretchLastSection(True)
        self.totals.setMaximumHeight(180)
        side.addWidget(self.totals)

        self.review_label = QLabel("Nothing to review")
        self.review_label.setWordWrap(True)
        side.addWidget(self.review_label)

        fix = QHBoxLayout()
        self.color_combo = QComboBox()
        self.color_combo.setEditable(True)
        self.color_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        names = []
        for color in self.palette.colors:
            self.color_combo.addItem(f"{color.id}  {color.name}", color.id)
            names.append(color.name)
        completer = QCompleter(names, self.color_combo)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.color_combo.setCompleter(completer)
        fix.addWidget(self.color_combo, stretch=1)
        save_color = QPushButton("Save color sample")
        save_color.clicked.connect(self._save_color_sample)
        fix.addWidget(save_color)
        side.addLayout(fix)

        side.addWidget(QLabel("List so far"))
        self.list_table = QTableWidget(0, 4)
        self.list_table.setHorizontalHeaderLabels(["Part", "Color", "Qty", "Cond"])
        self.list_table.horizontalHeader().setStretchLastSection(True)
        side.addWidget(self.list_table, stretch=1)
        body.addLayout(side, stretch=2)
        layout.addLayout(body, stretch=1)

        buttons = QHBoxLayout()
        for text, slot in (
            ("Identify part", self._identify),
            ("Count this pile", self._count),
            ("Empty table", self._empty_table),
            ("Grey card", self._grey_card),
            ("Show camera", self._show_camera),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        buttons2 = QHBoxLayout()
        for text, slot in (
            ("Add this pile to the list", self._add_batch),
            ("Export BSX", self._export),
            ("Clear list", self._clear_list),
            ("Check for updates", self._check_updates),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons2.addWidget(button)
        layout.addLayout(buttons2)

        self.update_btn = QPushButton("Update available - Install now")
        self.update_btn.clicked.connect(self._install_update)
        self.update_btn.hide()
        layout.addWidget(self.update_btn)
        self.status = QLabel("Point the camera at the table.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._refresh_list()

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def _open_camera(self, index: int) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = open_camera(index)
        if not cap.isOpened():
            cap.release()
            self._set_status("Camera not found. Pick another camera.")
            return
        self.cap = cap

    def _camera_changed(self) -> None:
        index = int(self.camera_combo.currentData())
        self.settings.camera_index = index
        self.settings.save()
        self._open_camera(index)

    def _condition_changed(self) -> None:
        self.settings.condition = str(self.condition_combo.currentData())
        self.settings.save()

    def _piece_area_changed(self) -> None:
        self.settings.piece_area = int(self.piece_area.value())
        self.settings.save()

    def _grab(self) -> np.ndarray | None:
        if self.cap is None or not self.cap.isOpened():
            self._set_status("Camera not found. Pick another camera.")
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._set_status("The camera did not return a photo. Try another camera.")
            return None
        return frame

    def _preview(self) -> None:
        if not self.live or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        self.frame = frame
        self._show(frame)

    def _show(self, image_bgr: np.ndarray) -> None:
        self.still = image_bgr
        pixmap = _bgr_to_pixmap(image_bgr)
        self.photo.setPixmap(
            pixmap.scaled(
                self.photo.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _image_point(self, pos) -> tuple[int, int] | None:
        if self.still is None or self.photo.pixmap() is None:
            return None
        pixmap = self.photo.pixmap()
        lw, lh = self.photo.width(), self.photo.height()
        pw, ph = pixmap.width(), pixmap.height()
        ox = (lw - pw) / 2
        oy = (lh - ph) / 2
        if pos.x() < ox or pos.y() < oy or pos.x() > ox + pw or pos.y() > oy + ph:
            return None
        height, width = self.still.shape[:2]
        x = int((pos.x() - ox) / pw * width)
        y = int((pos.y() - oy) / ph * height)
        return x, y

    def _photo_clicked(self, event) -> None:
        point = self._image_point(event.position())
        if point is None or self.still is None:
            return
        if self.count_result is not None and not self.live:
            self._select_piece(*point)
            return
        if not self.live:
            self._identify_crop(*point)

    def _run(self, fn, on_ok) -> None:
        if self._job is not None and self._job.isRunning():
            self._set_status("Still working on the last step.")
            return
        self._job = _Job(fn)
        self._job.ok.connect(on_ok)
        self._job.err.connect(self._set_status)
        self._job.start()

    def _clear_candidates(self) -> None:
        while self.candidates.count():
            item = self.candidates.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _identify(self) -> None:
        frame = self._grab()
        if frame is None:
            return
        self.live = False
        self.count_result = None
        self.raw = frame
        self._show(frame)
        self._set_status("Looking up the part...")
        self._run(lambda: self._identify_work(frame), self._show_candidates)

    def _identify_crop(self, x: int, y: int) -> None:
        if self.still is None:
            return
        crop = crop_around(self.still, x, y)
        self._set_status("Looking up the piece you clicked...")
        self._run(lambda: self._identify_work(crop), self._show_candidates)

    def _identify_work(self, image: np.ndarray):
        result = identify_image(image, top_k=5)
        packed = []
        for candidate in result.candidates[:5]:
            thumb = b""
            if candidate.image_url:
                try:
                    response = requests.get(candidate.image_url, timeout=4)
                    if response.ok:
                        thumb = response.content
                except requests.RequestException:
                    thumb = b""
            packed.append((candidate, thumb))
        return packed

    def _show_candidates(self, packed) -> None:
        self._clear_candidates()
        if not packed:
            self._set_status("No match. Type the part number.")
            return
        for candidate, thumb in packed:
            button = QPushButton(f"{candidate.part_id}  {candidate.name}  {candidate.score:.0%}")
            button.clicked.connect(
                lambda _checked=False, c=candidate: self._choose_part(c.part_id, c.name)
            )
            row = QHBoxLayout()
            holder = QWidget()
            holder.setLayout(row)
            if thumb:
                label = QLabel()
                pixmap = QPixmap()
                pixmap.loadFromData(thumb)
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaledToHeight(48, Qt.TransformationMode.SmoothTransformation))
                row.addWidget(label)
            row.addWidget(button, stretch=1)
            self.candidates.addWidget(holder)
        self._set_status("Pick the part, or type the number. Click a piece in the photo to try again.")

    def _choose_part(self, part_id: str, name: str) -> None:
        self.part_id = part_id.strip()
        self.part_name = name.strip()
        self.part_label.setText(f"Part: {self.part_id}  {self.part_name}".strip())
        self._set_status(f"Using part {self.part_id}. Dump the bin and count it.")

    def _use_typed_part(self) -> None:
        part_id = self.part_entry.text().strip()
        if not part_id:
            self._set_status("Type a part number first.")
            return
        self._choose_part(part_id, "")

    def _count(self) -> None:
        frame = self._grab()
        if frame is None:
            return
        self.live = False
        background = None
        path = self.settings.background_path
        if path.is_file():
            background = cv2.imread(str(path), cv2.IMREAD_COLOR)
        result = count_image(
            frame,
            self.palette,
            background_bgr=background,
            wb_gains=self.settings.gains_array(),
            piece_area=self.settings.piece_area or None,
        )
        self.raw = frame
        self.count_result = result
        self.selected_index = None
        self._show(draw_overlay(frame, result))
        self._fill_totals(result)
        reviews = result.review_indexes
        if not result.pieces:
            self._set_status("No pieces found. Use a plain mat, or take an empty-table photo.")
        elif reviews:
            self._set_status(
                f"Counted {sum(p.count for p in result.pieces)} pieces. "
                f"{len(reviews)} need a look. Click a box, pick the color, Save color sample."
            )
        else:
            self._set_status(f"Counted {sum(p.count for p in result.pieces)} pieces.")

    def _fill_totals(self, result: CountResult) -> None:
        totals = result.totals
        names = {piece.color_id: piece.color_name for piece in result.pieces}
        self.totals.setRowCount(len(totals))
        for row, (color_id, qty) in enumerate(sorted(totals.items(), key=lambda item: names.get(item[0], ""))):
            self.totals.setItem(row, 0, QTableWidgetItem(names.get(color_id, str(color_id))))
            self.totals.setItem(row, 1, QTableWidgetItem(str(qty)))
        if result.review_indexes:
            bits = []
            for index in result.review_indexes:
                piece = result.pieces[index]
                bits.append(f"#{index + 1} {piece.color_name} ({piece.reason})")
            self.review_label.setText("Check: " + "; ".join(bits))
        else:
            self.review_label.setText("Nothing to review")

    def _select_piece(self, x: int, y: int) -> None:
        if self.count_result is None:
            return
        for piece in self.count_result.pieces:
            px, py, pw, ph = piece.box
            if px <= x <= px + pw and py <= y <= py + ph:
                self.selected_index = piece.index
                idx = self.color_combo.findData(piece.color_id)
                if idx >= 0:
                    self.color_combo.setCurrentIndex(idx)
                self._set_status(f"Piece {piece.index + 1}: {piece.color_name}. Pick the right color and save.")
                return
        self._set_status("Click inside a box.")

    def _selected_color_id(self) -> int | None:
        data = self.color_combo.currentData()
        if data is None:
            text = self.color_combo.currentText().strip()
            for color in self.palette.colors:
                if text.lower() == color.name.lower() or text == str(color.id):
                    return color.id
            return None
        return int(data)

    def _save_color_sample(self) -> None:
        color_id = self._selected_color_id()
        if color_id is None:
            self._set_status("Pick a color from the list.")
            return
        if self.count_result is not None and self.selected_index is not None:
            piece = correct_piece(self.count_result, self.selected_index, color_id, self.palette)
            self.settings.add_learned(color_id, piece.lab)
            self.settings.save()
            self.palette.set_samples(self.settings.user_samples())
            rematch_unlocked(self.count_result, self.palette)
            if self.raw is not None:
                self._show(draw_overlay(self.raw, self.count_result))
            self._fill_totals(self.count_result)
            self._set_status(f"Saved. This piece is {piece.color_name}. The app will remember it.")
            return
        frame = self.raw if self.raw is not None and not self.live else self._grab()
        if frame is None:
            return
        self.live = False
        background = None
        if self.settings.background_path.is_file():
            background = cv2.imread(str(self.settings.background_path), cv2.IMREAD_COLOR)
        result = count_image(frame, self.palette, background_bgr=background, wb_gains=self.settings.gains_array())
        if not result.pieces:
            self._set_status("No piece in the photo to calibrate.")
            return
        piece = max(result.pieces, key=lambda item: item.area)
        self.settings.add_calibrated(color_id, piece.lab)
        self.settings.save()
        self.palette.set_samples(self.settings.user_samples())
        color = self.palette.by_id[color_id]
        self._set_status(f"Saved a lighting sample for {color.name}.")

    def _empty_table(self) -> None:
        frame = self._grab()
        if frame is None:
            return
        self.settings.directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.settings.background_path), frame)
        self._set_status("Empty table saved. Counts will ignore that backdrop.")

    def _grey_card(self) -> None:
        frame = self._grab()
        if frame is None:
            return
        gains = grey_card_gains(frame)
        self.settings.wb_gains = [float(x) for x in gains]
        self.settings.save()
        self._set_status("Grey card saved. New photos will use that white balance.")

    def _show_camera(self) -> None:
        self.live = True
        self.count_result = None
        self.selected_index = None
        self._set_status("Camera on.")

    def _add_batch(self) -> None:
        if not self.part_id:
            self._set_status("Identify the part first, or type the part number.")
            return
        if self.count_result is None or not self.count_result.pieces:
            self._set_status("Count the pile first.")
            return
        names = {piece.color_id: piece.color_name for piece in self.count_result.pieces}
        self.inventory.add_batch(
            self.part_id,
            self.part_name,
            self.count_result.totals,
            names,
            self.settings.condition,
        )
        self.settings.inventory = self.inventory.to_json()
        self.settings.save()
        self._refresh_list()
        reviews = len(self.count_result.review_indexes)
        note = f" Added with {reviews} pieces still marked to check." if reviews else ""
        self._set_status(f"Added to the list.{note} Dump the next bin when you are ready.")
        self.live = True
        self.count_result = None

    def _refresh_list(self) -> None:
        self.list_table.setRowCount(len(self.inventory.lines))
        for row, line in enumerate(self.inventory.lines):
            cond = "New" if line.condition == "N" else "Used"
            for col, text in enumerate((line.part_id, line.color_name or str(line.color_id), str(line.qty), cond)):
                self.list_table.setItem(row, col, QTableWidgetItem(text))

    def _export(self) -> None:
        if not self.inventory.lines:
            self._set_status("The list is empty.")
            return
        default = f"BrickTally-{datetime.now().strftime('%Y%m%d')}.bsx"
        path, _ = QFileDialog.getSaveFileName(self, "Export BSX", default, "BrickStore (*.bsx)")
        if not path:
            return
        text = to_bsx(self.inventory)
        validate_bsx(text)
        Path(path).write_text(text, encoding="utf-8")
        self._set_status(f"Saved {path}. Open it in BrickStore.")

    def _clear_list(self) -> None:
        if not self.inventory.lines:
            return
        answer = QMessageBox.question(self, "Clear list", "Clear the whole list?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.inventory.clear()
        self.settings.inventory = []
        self.settings.save()
        self._refresh_list()
        self._set_status("List cleared.")

    def _check_updates(self) -> None:
        if not repo_configured(UPDATE_REPO):
            self._set_status("Update check is not set up yet. You can keep counting.")
            return
        self._run(lambda: check_for_update(UPDATE_REPO), self._on_update_checked)

    def _on_update_checked(self, update) -> None:
        if update is None:
            self.update_btn.hide()
            self._set_status("No update. You are on the current version, or the check could not connect.")
            return
        self.pending_update = update
        self.update_btn.setText(f"Update available - Install now ({update.tag})")
        self.update_btn.show()
        self._set_status(f"Update {update.tag} is ready. Click Install now when you can pause.")

    def _install_update(self) -> None:
        update = self.pending_update
        if update is None:
            return
        dest = self.settings.directory / "updates" / update.asset_name
        self._set_status("Downloading the update...")
        self._run(lambda: download_verified(update, dest), self._on_downloaded)

    def _on_downloaded(self, path) -> None:
        message = spawn_updater(Path(path))
        if message:
            self._set_status(message)
            return
        self._set_status("Installing. BrickTally will close and reopen.")
        QApplication.instance().quit()

    def closeEvent(self, event) -> None:
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


def run() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication([])
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
