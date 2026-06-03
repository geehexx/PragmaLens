from importlib import util
from pathlib import Path
from typing import Any

import pytest


def _load_import_hygiene() -> Any:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "check_import_hygiene.py"
    spec = util.spec_from_file_location("check_import_hygiene", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load import hygiene script")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_module(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_import_hygiene_passes_for_top_level_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_hygiene = _load_import_hygiene()
    module_root = tmp_path / "src" / "pragmalens"
    _write_module(module_root, "clean.py", "import math\n\n\nVALUE = math.pi\n")
    monkeypatch.setattr(import_hygiene, "SCAN_ROOTS", [module_root])

    import_hygiene.main()


def test_import_hygiene_rejects_nested_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_hygiene = _load_import_hygiene()
    module_root = tmp_path / "src" / "pragmalens"
    _write_module(
        module_root,
        "bad.py",
        "def load_value() -> int:\n    import math\n\n    return int(math.pi)\n",
    )
    monkeypatch.setattr(import_hygiene, "SCAN_ROOTS", [module_root])

    with pytest.raises(SystemExit):
        import_hygiene.main()
