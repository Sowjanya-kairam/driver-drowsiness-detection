"""
Dataset validation script for driver drowsiness detection project.
Checks dataset structure, image integrity, and class distribution.
"""

import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from config import DEFAULT_DATA_DIR, CLASS_NAMES


def validate_dataset_structure(data_dir: Path) -> dict:
    """Validate the dataset folder structure."""
    report = {"structure": {}, "valid": True}
    
    # Check required directories
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    
    if not train_dir.exists():
        report["structure"]["train_dir"] = "❌ Missing"
        report["valid"] = False
    else:
        report["structure"]["train_dir"] = "✓ Found"
    
    if not test_dir.exists():
        report["structure"]["test_dir"] = "❌ Missing"
        report["valid"] = False
    else:
        report["structure"]["test_dir"] = "✓ Found"
    
    # Check subdirectories
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            report[f"{split}_classes"] = {}
            for class_name in CLASS_NAMES:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
                    report[f"{split}_classes"][class_name] = f"✓ {count} images"
                else:
                    report[f"{split}_classes"][class_name] = "❌ Missing"
                    report["valid"] = False
    
    return report


def validate_image_integrity(data_dir: Path) -> dict:
    """Check for corrupted or invalid images."""
    report = {"corrupted": [], "valid_count": 0, "invalid_count": 0}
    
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            for img_path in list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None or img.size == 0:
                        report["corrupted"].append(str(img_path))
                        report["invalid_count"] += 1
                    else:
                        report["valid_count"] += 1
                except Exception as e:
                    report["corrupted"].append(f"{img_path}: {str(e)}")
                    report["invalid_count"] += 1
    
    return report


def validate_image_dimensions(data_dir: Path, sample_size: int = 50) -> dict:
    """Check image dimensions consistency."""
    report = {"dimensions": defaultdict(list), "issues": []}
    
    image_count = 0
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            for img_path in list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")):
                if image_count >= sample_size:
                    break
                
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        key = f"{h}x{w}"
                        report["dimensions"][key].append(str(img_path.name))
                        image_count += 1
                except Exception as e:
                    report["issues"].append(f"{img_path.name}: {str(e)}")
    
    # Check if dimensions are consistent
    if len(report["dimensions"]) > 1:
        report["warning"] = f"⚠️  Multiple image dimensions found: {list(report['dimensions'].keys())}"
    else:
        report["status"] = "✓ All images have consistent dimensions"
    
    return report


def validate_class_distribution(data_dir: Path) -> dict:
    """Check class balance in the dataset."""
    report = {"train": {}, "test": {}, "balance_analysis": ""}
    
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
            report[split][class_name] = count
        
        if report[split]:
            total = sum(report[split].values())
            for class_name in CLASS_NAMES:
                percentage = (report[split][class_name] / total * 100) if total > 0 else 0
                report[split][f"{class_name}_pct"] = f"{percentage:.2f}%"
    
    # Analyze balance
    if report["train"]:
        drowsy_count = report["train"].get("drowsy", 0)
        non_drowsy_count = report["train"].get("non_drowsy", 0)
        if drowsy_count > 0 and non_drowsy_count > 0:
            ratio = max(drowsy_count, non_drowsy_count) / min(drowsy_count, non_drowsy_count)
            if ratio > 1.5:
                report["balance_analysis"] = f"⚠️  Class imbalance detected (ratio: {ratio:.2f}:1)"
            else:
                report["balance_analysis"] = "✓ Dataset is well-balanced"
    
    return report


def main():
    data_dir = Path(DEFAULT_DATA_DIR)
    
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    
    # 1. Structure validation
    print("\n1. STRUCTURE VALIDATION")
    print("-" * 60)
    structure_report = validate_dataset_structure(data_dir)
    for key, value in structure_report.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif key != "valid":
            print(f"{key}: {value}")
    
    # 2. Image integrity
    print("\n2. IMAGE INTEGRITY CHECK")
    print("-" * 60)
    integrity_report = validate_image_integrity(data_dir)
    print(f"Valid images: {integrity_report['valid_count']}")
    print(f"Invalid/Corrupted images: {integrity_report['invalid_count']}")
    if integrity_report["corrupted"]:
        print(f"\n❌ Corrupted files ({len(integrity_report['corrupted'])}):")
        for corrupted in integrity_report["corrupted"][:5]:  # Show first 5
            print(f"  - {corrupted}")
        if len(integrity_report["corrupted"]) > 5:
            print(f"  ... and {len(integrity_report['corrupted']) - 5} more")
    else:
        print("✓ All images are valid")
    
    # 3. Image dimensions
    print("\n3. IMAGE DIMENSIONS CHECK")
    print("-" * 60)
    dimensions_report = validate_image_dimensions(data_dir)
    if "warning" in dimensions_report:
        print(dimensions_report["warning"])
    else:
        print(dimensions_report.get("status", ""))
    print(f"\nDimensions found (sample of {50} images):")
    for dim, count in dimensions_report["dimensions"].items():
        print(f"  {dim}: {len(count)} images")
    
    # 4. Class distribution
    print("\n4. CLASS DISTRIBUTION")
    print("-" * 60)
    distribution_report = validate_class_distribution(data_dir)
    
    for split in ["train", "test"]:
        if distribution_report.get(split):
            print(f"\n{split.upper()} SET:")
            class_counts = {k: v for k, v in distribution_report[split].items() if not k.endswith("_pct")}
            for class_name, count in class_counts.items():
                pct = distribution_report[split].get(f"{class_name}_pct", "N/A")
                print(f"  {class_name}: {count} ({pct})")
    
    print(f"\n{distribution_report['balance_analysis']}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if structure_report["valid"] and integrity_report["invalid_count"] == 0:
        print("✓ Dataset validation PASSED")
        print("\nDataset is ready for training!")
    else:
        print("❌ Dataset validation FAILED")
        print("\nIssues found:")
        if not structure_report["valid"]:
            print("  - Missing required directories")
        if integrity_report["invalid_count"] > 0:
            print(f"  - {integrity_report['invalid_count']} corrupted/invalid images")
        print("\nPlease fix the issues before training.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
