"""Transfer only the nine checksummed, intentionally untracked frozen model files."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.freeze_model import CANDIDATES, FreezeError, artifact_names, read_json, record_hash


def expected_artifacts(root: Path) -> dict[str, str]:
    config = read_json(root / "config/model_config.json")
    if config.get("freeze_record_sha256") != record_hash(config):
        raise FreezeError("Frozen configuration checksum mismatch.")
    names = {path for name in CANDIDATES for path in artifact_names(name)}
    return {path: config["frozen_candidate"]["artifacts_sha256"][path] for path in sorted(names)}


def check_bytes(data: bytes, checksum: str, name: str) -> None:
    if hashlib.sha256(data).hexdigest() != checksum:
        raise FreezeError(f"Frozen model checksum mismatch: {name}")


def publish_new(temporary: Path, destination: Path) -> None:
    """Atomically publish on the same filesystem without replacing existing files."""
    os.link(temporary, destination)


def export_models(root: Path, archive: Path) -> int:
    expected = expected_artifacts(root)
    archive = archive.resolve()
    if archive.exists():
        raise FreezeError("Archive already exists; choose a new filename.")
    if archive.is_relative_to((root / "data/raw").resolve()):
        raise FreezeError("Never write an artifact bundle into immutable raw data.")
    content = {name: (root / name).read_bytes() for name in expected}
    for name, data in content.items():
        check_bytes(data, expected[name], name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=archive.parent, suffix=".zip.tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in content.items():
                bundle.writestr(name, data)
        publish_new(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    return len(content)


def restore_models(root: Path, archive: Path) -> int:
    expected = expected_artifacts(root)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise FreezeError("Bundle must contain exactly the nine expected model files; extra, duplicate or missing entries are rejected.")
        if any(member.file_size > 16 * 1024 * 1024 for member in bundle.infolist()):
            raise FreezeError("Bundle contains an unexpectedly large model artifact.")
        content = {name: bundle.read(name) for name in expected}
    destinations = {}
    # Validate the entire bundle and all existing destinations before publishing anything.
    for name, data in content.items():
        check_bytes(data, expected[name], name)
        destination = (root / name).resolve()
        if not destination.is_relative_to((root / "models").absolute()):
            raise FreezeError("Model destination escapes the local models directory.")
        if destination.exists():
            check_bytes(destination.read_bytes(), expected[name], name)
        else:
            destinations[destination] = data
    for destination, data in destinations.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=destination.parent, suffix=".model.tmp")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            publish_new(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return len(destinations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "restore"))
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        count = (export_models if args.action == "export" else restore_models)(ROOT, args.archive)
    except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.action}: {count} model files; frozen checksums verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
