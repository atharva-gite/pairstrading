import re
from pathlib import Path


def test_sidebar_slider_controls_are_defined_once_with_keys():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    content = app_path.read_text()

    slider_calls = re.findall(r"st\.slider\(", content)
    assert len(slider_calls) == 4, f"Expected 4 slider widgets, found {len(slider_calls)}"

    assert content.count("key=") >= 4, "Each sidebar control should have a unique Streamlit key"


def test_load_pair_uses_a_callback_instead_of_writing_widget_state_inline():
    content = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    assert 'st.button("Load pair", on_click=load_pair_into_ticket)' in content
    assert "st.session_state[\"asset_a\"] = ticker_label(chosen_a)" in content


def test_app_renders_without_running_a_ticket():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    assert [n.value for n in at.number_input if n.key == "max_leg_pct"] == [75]


def test_help_doc_explains_the_desk_for_a_general_reader():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "help.md").read_text()
    for phrase in ["market-neutral", "Cointegration", "Half-life", "z-score", "walk-forward", "AI Report"]:
        assert phrase in text
    for name in ["prices.png", "spread.png", "zscore.png", "returns.png"]:
        assert (root / "docs" / "assets" / "guide" / name).is_file()
    app = (root / "app.py").read_text()
    assert 'class="help-link" href="guide"' in app
    assert "AI Report" in app


def test_guide_page_renders():
    from streamlit.testing.v1 import AppTest

    guide = Path(__file__).resolve().parents[1] / "pages" / "guide.py"
    at = AppTest.from_file(str(guide), default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    rendered = " ".join(m.value for m in at.markdown)
    assert "market-neutral" in rendered
    assert "Coca-Cola" in rendered
