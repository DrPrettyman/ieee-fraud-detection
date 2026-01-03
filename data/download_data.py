#!/usr/bin/env python3
"""Download IEEE-CIS Fraud Detection dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set KAGGLE_API_TOKEN environment variable:
   - Go to kaggle.com/settings -> API Tokens -> Create New Token
   - Copy the KGAT_... token
   - export KAGGLE_API_TOKEN="KGAT_..."
3. Accept competition rules at:
   https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""

import os
import subprocess
import sys
from pathlib import Path


def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly configured."""
    # Check for new-style token (preferred)
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    # Check for legacy kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True

    print("ERROR: Kaggle API credentials not found.")
    print("\nTo set up Kaggle API (choose one):")
    print("\nOption 1 - API Token (recommended):")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Under 'API Tokens', click 'Create New Token'")
    print("3. Copy the KGAT_... token")
    print("4. Run: export KAGGLE_API_TOKEN='KGAT_...'")
    print("\nOption 2 - Legacy credentials:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Under 'Legacy API Credentials', create new key")
    print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
    return False


def check_kaggle_package() -> bool:
    """Check if kaggle package is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Run: pip install kaggle")
        return False


def download_competition_data(data_dir: Path) -> bool:
    """Download and extract competition data."""
    competition = "ieee-fraud-detection"

    print(f"Downloading {competition} dataset...")
    print("(This may take a few minutes - dataset is ~500MB)\n")

    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", competition,
                "-p", str(data_dir)
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            if "403" in result.stderr or "accept" in result.stderr.lower():
                print("ERROR: You must accept the competition rules first.")
                print("Visit: https://www.kaggle.com/competitions/ieee-fraud-detection/rules")
                print("Click 'I Understand and Accept' then re-run this script.")
                return False
            else:
                print(f"ERROR: {result.stderr}")
                return False

        print(result.stdout)

    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Ensure kaggle is installed and in PATH.")
        return False

    return True


def extract_data(data_dir: Path) -> bool:
    """Extract downloaded zip file."""
    zip_file = data_dir / "ieee-fraud-detection.zip"

    if not zip_file.exists():
        print(f"ERROR: Expected zip file not found: {zip_file}")
        return False

    print("Extracting data...")

    try:
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(data_dir)

        # Remove zip file after extraction
        zip_file.unlink()
        print("Extraction complete. Zip file removed.")

    except Exception as e:
        print(f"ERROR extracting: {e}")
        return False

    return True


def verify_data(data_dir: Path) -> bool:
    """Verify all expected files are present."""
    expected_files = [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
        "sample_submission.csv"
    ]

    print("\nVerifying downloaded files...")
    all_present = True

    for filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} - MISSING")
            all_present = False

    return all_present


def main():
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"

    # Ensure data directory exists
    data_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("IEEE-CIS Fraud Detection Dataset Downloader")
    print("=" * 50 + "\n")

    # Check prerequisites
    if not check_kaggle_package():
        sys.exit(1)

    if not check_kaggle_setup():
        sys.exit(1)

    # Check if data already exists
    if (data_dir / "train_transaction.csv").exists():
        print("Data already downloaded. To re-download, delete files in data/ first.")
        verify_data(data_dir)
        sys.exit(0)

    # Download
    if not download_competition_data(data_dir):
        sys.exit(1)

    # Extract
    if not extract_data(data_dir):
        sys.exit(1)

    # Verify
    if verify_data(data_dir):
        print("\n✓ Dataset ready! You can now start with the EDA notebook.")
    else:
        print("\n⚠ Some files may be missing. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
