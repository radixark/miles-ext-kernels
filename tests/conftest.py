"""Every test leaves the Cake runtime as it found it: default configuration, empty workspace pool."""

import pytest

from miles_ext_kernels import cake


@pytest.fixture(autouse=True)
def _reset_cake_runtime():
    yield
    cake.configure()
    cake.release()


@pytest.fixture
def pooled():
    """The workspace pool bounded at 1 GiB for this test (``helpers.POOLED``; the default, 0, keeps nothing). A
    configure call of the test replaces the whole configuration: pass ``**POOLED`` to it as well."""
    from cake.helpers import POOLED

    cake.configure(**POOLED)
    return POOLED


@pytest.fixture
def synth(monkeypatch):
    """The synthetic op ``synth`` registered for this test (``tests/cake/_runtime/synthetic.py``)."""
    from cake._runtime import synthetic

    return synthetic.install(monkeypatch)
