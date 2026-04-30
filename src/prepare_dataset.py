import argparse
import shutil
import zipfile
from pathlib import Path


def extract_dataset(zip_path: Path, output_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)

    macosx = output_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)

    dataset_dir = output_dir / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError("Archive did not produce a dataset/ directory.")

    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the driver drowsiness dataset.")
    parser.add_argument("--zip", default="dataset.zip", help="Path to dataset zip archive.")
    parser.add_argument("--output", default=".", help="Directory where dataset/ should be created.")
    args = parser.parse_args()

    dataset_dir = extract_dataset(Path(args.zip), Path(args.output))
    print(f"Dataset ready at: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
