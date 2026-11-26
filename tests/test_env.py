import pytest
from conftest import INSTALL_CMD, require_pkg


@pytest.mark.parametrize("name", ["pyarrow", "lightgbm", "sgfmill"])
def test_required_packages_installed(name):
    require_pkg(name)


def test_install_command_text():
    assert "pip install pyarrow lightgbm sgfmill" in INSTALL_CMD
