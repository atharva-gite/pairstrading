import re
from pathlib import Path


def test_sidebar_slider_controls_are_defined_once_with_keys():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    content = app_path.read_text()

    slider_calls = re.findall(r"st\.slider\(", content)
    assert len(slider_calls) == 4, f"Expected 4 slider widgets, found {len(slider_calls)}"

    assert content.count("key=") >= 4, "Each sidebar control should have a unique Streamlit key"


def test_app_renders_without_running_a_ticket():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    assert [n.value for n in at.number_input if n.key == "max_leg_pct"] == [75]
