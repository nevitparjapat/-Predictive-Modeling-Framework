from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_end_to_end():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["① Your data", "② Train & compare", "③ Forecast", "④ Report card"]

    at.button[0].click().run()          # "Train models"
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Typical miss" in labels and "Versus the average season" in labels
    assert len(at.slider) == 9

    at.selectbox[0].select("Drought season").run()
    assert not at.exception
