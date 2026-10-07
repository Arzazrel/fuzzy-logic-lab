"""
Test: l'implementazione from-scratch e quella con scikit-fuzzy devono
produrre risultati coerenti (entro la tolleranza dovuta alla discretizzazione).

Esegui con:  pytest tests/  (dalla cartella 01_mamdani_fis)
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import example_tourists as scratch
import example_tourists_skfuzzy as sk
from membership import trimf


TEST_POINTS = [(19, 60), (35, 90), (5, 20), (25, 55), (30, 70), (12, 40)]


@pytest.fixture(scope="module")
def systems():
    return scratch.build_system(), sk.build_system()


@pytest.mark.parametrize("temp,sun", TEST_POINTS)
def test_scratch_vs_skfuzzy(systems, temp, sun):
    fis_scratch, sim_sk = systems
    y_scratch = fis_scratch.infer({"temperature": temp, "sunshine": sun})
    y_sk = sk.predict(sim_sk, temp, sun)
    # tolleranza 2%: le differenze derivano dalla discretizzazione
    assert abs(y_scratch - y_sk) < 2.0, (
        f"Divergenza a temp={temp}, sun={sun}: "
        f"scratch={y_scratch:.2f}, skfuzzy={y_sk:.2f}"
    )


def test_triangular_mf_properties():
    """La MF triangolare deve valere 1 al vertice e 0 ai piedi."""
    x = np.arange(0, 41, 1.0)
    mf = trimf(x, (10, 20, 30))
    assert np.isclose(mf[x == 20][0], 1.0)
    assert np.isclose(mf[x == 10][0], 0.0)
    assert np.isclose(mf[x == 30][0], 0.0)
    assert np.all((mf >= 0) & (mf <= 1))


def test_output_range():
    """L'output deve sempre restare nell'universo [0, 100]."""
    fis = scratch.build_system()
    for temp in range(0, 41, 5):
        for sun in range(0, 101, 10):
            y = fis.infer({"temperature": temp, "sunshine": sun})
            assert 0 <= y <= 100
