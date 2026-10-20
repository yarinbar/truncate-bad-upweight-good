from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PINS = {
    "torch": "2.4.0+cu121", "transformers": "4.51.3", "trl": "0.13.0", "accelerate": "1.2.1",
    "deepspeed": "0.15.4", "vllm": "0.6.1", "datasets": "3.2.0", "hydra-core": "1.3.2",
    "omegaconf": "2.3.0", "mpmath": "1.3.0", "alpaca_eval": "0.6.6", "numpy": "1.26.4",
    "scikit-learn": "1.5.2", "pandas": "2.2.2", "huggingface-hub": "0.36.2", "peft": "0.14.0",
}


def pins():
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name, version = line.split("==")
        out[name.lower().replace("_", "-")] = version
    return out


def test_requirements_pin_the_paper_environment():
    p = pins()
    for name, version in REQUIRED_PINS.items():
        assert p[name.lower().replace("_", "-")] == version, name


def test_requirements_have_no_local_or_editable_entries():
    text = (ROOT / "requirements.txt").read_text()
    assert "file://" not in text and "-e " not in text
    assert not any(line.lower().startswith(("qrpo==", "truncate-bad-upweight-good==")) for line in text.splitlines())


def test_flash_attn_installed_separately():
    assert "flash-attn" not in pins() and "flash_attn" not in pins()
    text = (ROOT / "install.sh").read_text()
    assert "flash-attn==2.8.3" in text
    # The prebuilt wheel is pinned by digest (pip verifies the #sha256= fragment).
    assert ("flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            "#sha256=f6e2c4172c98a8225b45424dd87ab43224df878cfb4148d679e2c98a7a7dd24d") in text
