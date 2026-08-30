from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_video_uses_the_prebuilt_signed_bundle() -> None:
    config = (ROOT / "app/frontend/playwright.demo.config.ts").read_text()
    script = (ROOT / "scripts/record_demo_video.sh").read_text()

    assert "npm run build" not in config
    assert '"$repo_root/scripts/release.py" \\' in script
    assert '--root "$repo_root" --verify' in script
    assert script.index("scripts/release.py") < script.index("playwright test")
