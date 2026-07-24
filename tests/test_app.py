from pathlib import Path
import re


def test_sidebar_slider_controls_are_defined_once_with_keys():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    content = app_path.read_text()

    slider_calls = re.findall(r"st\.slider\(", content)
    assert len(slider_calls) == 4, f"Expected 4 slider widgets, found {len(slider_calls)}"

    assert content.count("key=") >= 4, "Each sidebar control should have a unique Streamlit key"
