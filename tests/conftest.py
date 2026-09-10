import pytest

from flycraftax.data import load_connectome


@pytest.fixture(scope="session")
def conn():
    return load_connectome(threshold=5)
