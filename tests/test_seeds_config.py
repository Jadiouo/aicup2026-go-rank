import pytest
import yaml

from gogame.config import config_hash, load_config
from gogame.errors import ConfigError
from gogame.seeds import derive_seed


def test_derive_seed_constants():
    assert derive_seed(2026, "split") == 1522991044303194512
    assert derive_seed(2026, "val_questions") == 6928604500976227792


def test_derive_seed_streams_differ():
    assert derive_seed(2026, "split") != derive_seed(2026, "train_bags")
    assert derive_seed(2026, "split") == derive_seed(2026, "split")
    assert derive_seed(2026, "a", "b") != derive_seed(2026, "a")
    assert 0 <= derive_seed(1, "x") < 2**63


def test_config_hash_stable_and_key_order_independent():
    a = {"seed": 2026, "lr": 0.1, "nested": {"x": 1, "y": 2}}
    b = {"nested": {"y": 2, "x": 1}, "lr": 0.1, "seed": 2026}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 16
    c = dict(a, lr=0.2)
    assert config_hash(a) != config_hash(c)


def test_load_config_missing_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"lr": 0.1}))
    with pytest.raises(ConfigError):
        load_config(p)


def test_load_config_ok_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"seed": 2026, "lr": 0.1}))
    assert load_config(p)["lr"] == 0.1
    assert load_config(p, {"lr": 0.5})["lr"] == 0.5


def test_constants():
    from gogame import constants as C
    assert C.RANKS == ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
    assert C.RANK_TO_ID["A"] == 3 and C.ID_TO_RANK[9] == "6D"
    assert (C.BOARD, C.SEED, C.OPEN_PLIES, C.OPEN_OWN_MOVES, C.BIGRAM_BUCKETS) == (19, 2026, 24, 12, 1024)
