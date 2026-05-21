from pathlib import Path

REQUIRED_PATHS = [
    "README.md",
    "REPORT_TEMPLATE.md",
    "requirements.txt",
    "configs/baseline_vit_head_only.json",
    "configs/debug_smoke.json",
    "docs/DATASET_GUIDE.md",
    "docs/LAB_GUIDE_LAB5.md",
    "docs/WANDB_GUIDE.md",
    "src/__init__.py",
    "src/dataset.py",
    "src/model.py",
    "src/prepare_subset.py",
    "src/train.py",
    "src/utils.py",
    "ci/smoke_train.py",
]

def main() -> None:
    missing = [p for p in REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"- {p}")
        raise SystemExit(1)
    print("Structure check passed.")

if __name__ == "__main__":
    main()
