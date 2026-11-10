"""Build a small allowlisted upload bundle; no credentials, environments, or runs."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = sorted((ROOT / "illusion_agent").glob("*.py"))
    files += sorted((ROOT / "illusion_agent/prompts").glob("*.txt"))
    files += sorted((ROOT / "data/ambibench_debug").glob("*.json*"))
    files += sorted((ROOT / "data/ambibench_debug/images").glob("*"))
    files += [ROOT / "docs/05_free_colab.md", ROOT / "docs/04_research_contribution.md"]
    destination = ROOT / "deliverables/bica_colab_bundle.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Bundle failed CRC validation")
    print(destination, destination.stat().st_size, "bytes")


if __name__ == "__main__":
    build()
