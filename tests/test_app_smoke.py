"""Rauchtests: die App läuft für Default und jedes Preset ohne Exception durch (fängt u.a.
StreamlitDuplicateElementId bei st.plotly_chart und Slider-Grenzfälle)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cn_constants as C

APP_TIMEOUT = 120
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _run():
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT)
    at.run()
    return at


def test_default_run_without_exception():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("index,name", list(enumerate(C.PRESETS)))
def test_each_preset_runs_without_exception(index, name):
    at = _run()
    at.button[index].click().run()
    assert not at.exception, f"{name}: {[e.value for e in at.exception]}"
    assert at.session_state["env_mode_radio"] == C.PRESETS[name]["env_mode"]
    assert at.session_state["episodes_slider"] == C.PRESETS[name]["episodes"]


def test_switching_to_random_mode_keeps_sigma_value():
    at = _run()
    at.session_state["sigma_slider"] = 0.45
    at.radio(key="env_mode_radio").set_value(C.ENV_RANDOM).run()
    assert not at.exception
    assert at.session_state["sigma_slider"] == 0.45
    at.radio(key="env_mode_radio").set_value(C.ENV_RECURRING).run()
    assert not at.exception
    assert at.session_state["sigma_slider"] == 0.45


def test_small_instance_edge_cases():
    at = _run()
    at.slider(key="n_jobs_slider").set_value(C.N_JOBS_MIN)
    at.slider(key="n_agents_slider").set_value(C.N_AGENTS_MIN)
    at.select_slider(key="episodes_slider").set_value(100).run()
    assert not at.exception, [e.value for e in at.exception]
