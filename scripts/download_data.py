"""Télécharge les données brutes du projet.

- gretelai/synthetic_text_to_sql → data/raw/gretel/   (SFT)
- xlangai/spider                 → data/raw/spider/   (questions + SQL de référence)
- bases SQLite officielles Spider → data/spider/       (exécution des requêtes, eval + reward)

Usage : uv run scripts/download_data.py
"""

import shutil
import zipfile
from pathlib import Path

import gdown
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SPIDER_DIR = ROOT / "data" / "spider"

# Lien Google Drive donné sur https://yale-lily.github.io/spider
SPIDER_ZIP_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def download_hf_datasets() -> None:
    for name, target in [
        ("gretelai/synthetic_text_to_sql", RAW / "gretel"),
        ("xlangai/spider", RAW / "spider"),
    ]:
        if target.exists():
            print(f"✓ {name} déjà présent")
            continue
        print(f"↓ {name}")
        load_dataset(name).save_to_disk(target)


def download_spider_databases() -> None:
    if (SPIDER_DIR / "database").exists():
        print("✓ bases Spider déjà présentes")
        return
    zip_path = RAW / "spider_data.zip"
    if not zip_path.exists():
        print("↓ bases SQLite Spider (Google Drive)")
        gdown.download(id=SPIDER_ZIP_ID, output=str(zip_path), quiet=False)
    extract_dir = RAW / "spider_extract"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    # Le zip contient un dossier racine (spider_data/) avec database/, test_database/, *.json
    inner = next(p for p in extract_dir.iterdir() if p.is_dir() and not p.name.startswith("__"))
    shutil.rmtree(SPIDER_DIR, ignore_errors=True)
    shutil.move(str(inner), SPIDER_DIR)
    shutil.rmtree(extract_dir)
    print(f"✓ bases Spider extraites dans {SPIDER_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    download_hf_datasets()
    download_spider_databases()
