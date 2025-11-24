import os
import re
import csv
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image
from rapidfuzz import process as rf_process, fuzz as rf_fuzz


# ---------------------------
# Logging setup
# ---------------------------
def setup_logging(log_dir: Path) -> None:
    """
    Configure logging to both file and console.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


# ---------------------------
# Config loading
# ---------------------------
def load_config(config_path: Path) -> dict:
    """
    Load configuration from JSON file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# ---------------------------
# Tesseract initialization
# ---------------------------
def init_tesseract(tesseract_cmd: str) -> None:
    """
    Initialize Tesseract command path if provided.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# ---------------------------
# Event name normalization
# ---------------------------
def normalize_event_name(raw_name: str, canonical_events: dict, min_similarity: int = 70) -> str:
    """
    Normalize event folder name to canonical event name using abbreviations and fuzzy matching.
    """
    candidates = []
    for canonical, synonyms in canonical_events.items():
        candidates.append(canonical)
        candidates.extend(synonyms)

    match = rf_process.extractOne(raw_name, candidates, scorer=rf_fuzz.QRatio)
    if match:
        best_match, score, _ = match
        if score >= min_similarity:
            for canonical, synonyms in canonical_events.items():
                if best_match == canonical or best_match in synonyms:
                    return canonical
    return raw_name


# ---------------------------
# Utility: split folder name into event and date range
# ---------------------------
def split_event_and_date(folder_name: str, delimiter: str) -> Tuple[str, str]:
    """
    Split folder name into event and date range using the last delimiter occurrence.
    """
    if delimiter not in folder_name:
        logging.warning(f"No delimiter '{delimiter}' in folder name: {folder_name}. Using entire name as event.")
        return folder_name, "unknown"
    idx = folder_name.rfind(delimiter)
    event = folder_name[:idx].strip()
    date_range = folder_name[idx+1:].strip()
    return event or "unknown_event", date_range or "unknown"


# ---------------------------
# Image loading with robust handling
# ---------------------------
def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load image with OpenCV, fallback to PIL if needed.
    """
    if not image_path.exists():
        logging.error(f"Image not found: {image_path}")
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            return None
    return img


# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(img: np.ndarray, do_gray: bool, do_thresh: bool, thresh_val: int) -> np.ndarray:
    """
    Apply grayscale and threshold preprocessing to improve OCR accuracy.
    """
    proc = img
    if do_gray:
        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    if do_thresh:
        if len(proc.shape) == 3:
            proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        _, proc = cv2.threshold(proc, thresh_val, 255, cv2.THRESH_BINARY)
    return proc


# ---------------------------
# OCR extraction
# ---------------------------
def ocr_extract_text(img: np.ndarray, lang: str, preprocess_cfg: dict, retry_with_pre: bool) -> str:
    """
    Run OCR on image, retry with preprocessing if needed.
    """
    try:
        text = pytesseract.image_to_string(img, lang=lang)
        if text.strip():
            return text
    except Exception as e:
        logging.warning(f"OCR raw attempt failed: {e}")

    if retry_with_pre:
        try:
            proc = preprocess(
                img,
                preprocess_cfg.get("grayscale", True),
                preprocess_cfg.get("threshold", True),
                int(preprocess_cfg.get("threshold_value", 150))
            )
            text = pytesseract.image_to_string(proc, lang=lang)
            return text or ""
        except Exception as e:
            logging.error(f"OCR preprocess attempt failed: {e}")

    return ""


# ---------------------------
# Regex parsing
# ---------------------------
LINE_PATTERN = re.compile(r"\[([A-Za-z]{3})\]\s*(.+?)\s+([\d,]+)")




def parse_entries(text: str) -> List[Tuple[str, str, str, int]]:
    """
    Parse OCR text into structured leaderboard entries.
    """
    entries = []
    for match in LINE_PATTERN.finditer(text):
        tag, name, points_raw = match.groups()
        name_clean = " ".join(name.strip().split())
        try:
            points_num = int(points_raw.replace(",", ""))
        except ValueError:
            points_num = -1
        entries.append((tag, name_clean, points_raw, points_num))
    return entries


# ---------------------------
# CSV writing
# ---------------------------
def write_csv(rows: List[Tuple[str, str, str, int, str]], header: List[str], out_path: Path) -> None:
    """
    Write extracted data to CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------
# Process a single event folder
# ---------------------------

def process_event_folder(folder_path: Path, cfg: dict) -> Optional[Path]:
    """
    Process all screenshots in one event folder and save to CSV.
    """
    image_exts = {ext.lower() for ext in cfg["files"]["image_extensions"]}
    header = cfg["files"]["csv_header"]
    base_output_dir = Path(cfg["paths"]["base_output_dir"])
    delimiter = cfg["naming"]["date_range_delimiter"]
    csv_name_format = cfg["naming"]["csv_name_format"]

    event_mapping_cfg = cfg["event_mapping"]
    canonical_events = event_mapping_cfg["canonical_events"]
    min_sim = event_mapping_cfg.get("min_similarity", 70)

    event_raw, date_range = split_event_and_date(folder_path.name, delimiter)
    event_canonical = normalize_event_name(event_raw, canonical_events, min_sim)

    out_event_dir = base_output_dir / event_canonical
    csv_name = csv_name_format.format(date_range=date_range)
    out_csv_path = out_event_dir / csv_name

    lang = cfg["ocr"]["languages"]
    retry_pre = bool(cfg["ocr"].get("retry_with_preprocessing", True))
    preprocess_cfg = cfg["ocr"].get("preprocess", {})

    all_rows = []
    for entry in sorted(folder_path.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in image_exts:
            continue
        img = load_image(entry)
        if img is None:
            continue
        text = ocr_extract_text(img, lang, preprocess_cfg, retry_pre)
        matches = parse_entries(text)
        for tag, name, points_raw, points_num in matches:
            all_rows.append((tag, name, points_raw, points_num, entry.name))

    if not all_rows:
        logging.info(f"No data extracted for folder: {folder_path}")
        return None

    write_csv(all_rows, header, out_csv_path)
    logging.info(f"Saved CSV for '{event_raw}' (canonical: '{event_canonical}') -> {out_csv_path}")
    return out_csv_path


# ---------------------------
# Process all event folders
# ---------------------------
def process_all_events(cfg: dict) -> None:
    """
    Loop through all event folders and process them.
    """
    base_input_dir = Path(cfg["paths"]["base_input_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    setup_logging(log_dir)
    init_tesseract(cfg["paths"].get("tesseract_cmd", ""))

    if not base_input_dir.exists():
        logging.error(f"Input directory not found: {base_input_dir}")
        return

    for folder in sorted(base_input_dir.iterdir()):
        if folder.is_dir():
            process_event_folder(folder, cfg)


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    config_path = Path("config.json")
    config = load_config