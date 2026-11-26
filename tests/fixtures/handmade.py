"""Hand-written SGF table (tdd 2.2). All expectations are hand-computed in the tests."""
from helpers import sgf

HANDMADE: dict[str, str] = {
    "capture_single": sgf("B:ji W:jj B:ij W:aa B:kj W:ab B:jk"),
    "capture_corner": sgf("B:ba W:aa B:ab"),
    "capture_multi": sgf("B:ji W:jj B:ki W:kj B:li W:lj B:jk W:aa B:kk W:ab B:lk W:ac B:ij W:ad B:mj"),
    "ko": sgf("B:ci W:cj B:ck W:di B:bj W:dk B:sa W:ej B:dj"),
    "ko_recapture": sgf("B:ci W:cj B:ck W:di B:bj W:dk B:sa W:ej B:dj W:cj"),
    "suicide": sgf("B:sa W:ba B:sb W:ab B:aa"),
    "pass_variants": "(;SZ[19];B[pd];W[];B[dd];W[tt];B[pp])",
    "transpose_probe": sgf("B:pd W:aj"),
    "edge_coords": sgf("B:ss W:tt"),
    "bad_coord": sgf("B:ta"),
    "setup_stones": "(;SZ[19]AB[dd][pp];W[qq])",
    "mini_B": sgf("B:pd W:dp B:pp W:dd B:kj"),
    "white_first": sgf("W:pd B:dp W:pp B:dd W:kj"),
    "nonalt": sgf("B:pd B:dd"),
    "empty_game": "(;SZ[19])",
    "bad_size": "(;SZ[13];B[dd])",
}
