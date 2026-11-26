"""Thin driver for `katago analysis` + SGF/GTP helpers. Runs on Colab VM (and locally for parsing)."""
import glob, json, math, os, re, subprocess, threading, time

COLS = "ABCDEFGHJKLMNOPQRST"
PROFILES = ["rank_12k", "rank_9k", "rank_6k", "rank_3k", "rank_1k", "rank_1d",
            "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d"]


def rc_to_gtp(r, c):
    return COLS[c] + str(19 - r)


def parse_sgf_moves(sgf):
    """-> list of (color 'B'/'W', (row,col) or None for pass). SGF 'pd' = col 'p'(15), row 'd'(3)."""
    out = []
    for col, xy in re.findall(r";([BW])\[([a-z]*)\]", sgf):
        if len(xy) != 2 or xy == "tt":
            out.append((col, None))
        else:
            out.append((col, (ord(xy[1]) - 97, ord(xy[0]) - 97)))
    return out


# ------------------------------------------------------------------ format normalisation (handicap / missing SZ)
# Q13 of the official FAQ (2026-10-29, not final).  The pure-python twin of src/gogame/sgf.py:normalize_sgf; this file ships
# alone to school/Colab so it cannot import gogame.  tests/test_sgf_format.py asserts both implementations agree.
HANDICAP_STONES = {
    2: ["pd", "dp"], 3: ["pd", "dp", "dd"], 4: ["pd", "dp", "dd", "pp"], 5: ["pd", "dp", "dd", "pp", "jj"],
    6: ["pd", "dp", "dd", "pp", "dj", "pj"], 7: ["pd", "dp", "dd", "pp", "dj", "pj", "jj"],
    8: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp"], 9: ["pd", "dp", "dd", "pp", "dj", "pj", "jd", "jp", "jj"],
}
COLOR_SEMANTICS = ("as_annotated", "actual")
DEFAULT_COLOR_SEMANTICS = "as_annotated"
KNOWN_ROOT_PROPS = ("SZ", "HD", "CA", "RE")
_PROP_RE = re.compile(r"([A-Za-z]+)((?:\s*\[(?:\\.|[^\]\\])*\])+)", re.S)
_VAL_RE = re.compile(r"\[((?:\\.|[^\]\\])*)\]", re.S)
_SZ_ANY_RE = re.compile(r"(?<![A-Za-z])SZ\[([^\]]*)\]")


def get_color_semantics(value=None):
    v = value or os.environ.get("GOGAME_COLOR_SEMANTICS") or DEFAULT_COLOR_SEMANTICS
    if v not in COLOR_SEMANTICS:
        raise ValueError(f"color_semantics must be one of {COLOR_SEMANTICS}, got {v!r}")
    return v


def root_props(sgf):
    """Root node properties -> {NAME: [values]}.  Root node = text between the first ';' and the next ';' outside brackets."""
    if not isinstance(sgf, str):
        return {}
    i = sgf.find(";")
    if i < 0:
        return {}
    j, inside, esc = i + 1, False, False
    while j < len(sgf):
        ch = sgf[j]
        if esc:
            esc = False
        elif inside and ch == "\\":
            esc = True
        elif ch == "[":
            inside = True
        elif ch == "]":
            inside = False
        elif ch == ";" and not inside:
            break
        j += 1
    out = {}
    for m in _PROP_RE.finditer(sgf[i + 1:j]):
        out.setdefault(m.group(1), []).extend(_VAL_RE.findall(m.group(2)))
    return out


def _sgf_pt(xy):
    return (ord(xy[1]) - 97, ord(xy[0]) - 97)


def first_illegal(moves, stones=()):
    """Index of the first move that is occupied or suicide (simple board replay incl. captures, handicap stones first); None if all
    fine.  Ko is NOT checked (the engine-side error handling covers it).  Passes (rc None) are always legal."""
    g = {}
    for c, (r, q) in stones:
        g[(r, q)] = c

    def nb(r, q):
        for a, b in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1)):
            if 0 <= a < 19 and 0 <= b < 19:
                yield (a, b)

    def group(p):
        col, seen, st, lib = g[p], {p}, [p], set()
        while st:
            x = st.pop()
            for y in nb(*x):
                if y not in g:
                    lib.add(y)
                elif g[y] == col and y not in seen:
                    seen.add(y); st.append(y)
        return seen, lib

    for i, (c, rc) in enumerate(moves):
        if rc is None:
            continue
        if rc in g:
            return i
        g[rc] = c
        for y in nb(*rc):
            if y in g and g[y] != c and not group(y)[1]:
                for z in group(y)[0]:
                    del g[z]
        if rc in g and not group(rc)[1]:
            return i                              # suicide
    return None


class NormGame:
    """moves: [(col, rc|None)] already colour-swapped for HD>=2; eff_color: 0/1 side of the annotated player inside `moves`;
    initial_stones: [(col, rc)] (HD>=2); komi: None = keep the pinned komi, else per-game komi; skip: reason or None."""
    __slots__ = ("moves", "eff_color", "initial_stones", "initial_player", "komi", "hd", "skip", "root", "truncated")

    def __init__(self, moves=None, eff_color=0, initial_stones=None, initial_player=None, komi=None, hd=0, skip=None, root=None):
        self.moves, self.eff_color, self.initial_stones = moves or [], eff_color, initial_stones or []
        self.initial_player, self.komi, self.hd, self.skip, self.root = initial_player, komi, hd, skip, root or {}
        self.truncated = False      # True: moves were cut before the first illegal move (occupied point / suicide)


def normalize_game(sgf, color, color_semantics=None):
    """Current format (SZ[19] or no SZ, no HD / HD[0]) -> moves identical to parse_sgf_moves(sgf), eff_color == color, komi None.
    AB/AW in the SGF, SZ != 19, unusable HD, empty text -> skip reason (never silent)."""
    cs = get_color_semantics(color_semantics)
    color = int(color)
    if not isinstance(sgf, str) or not sgf.strip():
        return NormGame(eff_color=color, skip="empty_sgf")
    if ";" not in sgf:
        return NormGame(eff_color=color, skip="no_nodes")
    root = root_props(sgf)
    sz = root.get("SZ") or _SZ_ANY_RE.findall(sgf)[:1]
    if sz:
        m = re.fullmatch(r"\s*(\d+)\s*(?::\s*(\d+)\s*)?", sz[0])
        if not m or int(m.group(1)) != 19 or (m.group(2) is not None and int(m.group(2)) != 19):
            return NormGame(eff_color=color, skip="bad_size", root=root)
    if "AB[" in sgf or "AW[" in sgf:
        return NormGame(eff_color=color, skip="has_setup", root=root)
    hd = 0
    if "HD" in root:
        try:
            hd = int(root["HD"][0].strip())
        except ValueError:
            return NormGame(eff_color=color, skip="bad_hd", root=root)
        if hd < 0 or hd > 9:
            return NormGame(eff_color=color, skip="bad_hd", root=root)
    mv = parse_sgf_moves(sgf)
    if hd <= 1:
        ng = NormGame(mv, color, None, None, None if hd == 0 else 0.0, hd, None, root)
    else:
        mv = [("W" if c == "B" else "B", rc) for c, rc in mv]
        stones = [("B", _sgf_pt(x)) for x in HANDICAP_STONES[hd]]
        eff = 1 - color if cs == "as_annotated" else color
        ng = NormGame(mv, eff, stones, "W", 0.0, hd, None, root)
    # Moves from the first illegal one on are dropped (reason `truncated_illegal`, counted by callers): the own moves before it are
    # still valid training signal, and only the first 60 own moves are ever queried, so later trouble costs nothing.
    bad = first_illegal(ng.moves, ng.initial_stones)
    if bad is not None:
        ng.moves, ng.truncated = ng.moves[:bad], True
    return ng


def stones_gtp(stones):
    return [[c, rc_to_gtp(*rc)] for c, rc in stones]


def moves_gtp(moves):
    return [[c, "pass" if rc is None else rc_to_gtp(*rc)] for c, rc in moves]


def move_idx(rc):  # index into 362 policy (row*19+col, row 0 = top), pass = 361
    return 361 if rc is None else rc[0] * 19 + rc[1]


class Engine:
    def __init__(self, exe, model, human, cfg, overrides=(), log="/content/katago_stderr.log", cu12=None, env_extra=None):
        cmd = [exe, "analysis", "-config", cfg, "-model", model]
        if human:
            cmd += ["-human-model", human]
        for o in overrides:
            cmd += ["-override-config", o]
        self.logf = open(log, "w")
        env = dict(os.environ)
        cu12 = cu12 or os.environ.get("CU12_DIR", "/content/cu12")
        extra = ":".join(glob.glob(cu12 + "/nvidia/*/lib"))  # CUDA12 libs (Colab VM has CUDA13)
        env.update(env_extra or {})
        env["LD_LIBRARY_PATH"] = extra + ":" + env.get("LD_LIBRARY_PATH", "")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.ready = threading.Event()
        threading.Thread(target=self._err, daemon=True).start()
        for _ in range(600):
            if self.ready.wait(1):
                break
            if self.p.poll() is not None:
                time.sleep(1)
                raise RuntimeError("KataGo exited at startup: " + open(log).read()[-800:])
        else:
            raise RuntimeError("KataGo did not start; see " + log)

    def _err(self):
        for line in self.p.stderr:
            self.logf.write(line); self.logf.flush()
            if "ready to begin handling requests" in line:
                self.ready.set()

    def run(self, queries):
        """Send all queries, collect all per-turn responses. -> (list of responses, wall seconds)."""
        expect = sum(len(q["analyzeTurns"]) for q in queries)
        remaining = {q["id"]: len(q["analyzeTurns"]) for q in queries}
        self.last_errors = {}
        def w():
            for q in queries:
                self.p.stdin.write(json.dumps(q) + "\n")
            self.p.stdin.flush()
        t0 = time.time()
        th = threading.Thread(target=w, daemon=True); th.start()
        res = []; got = 0
        while got < expect:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            r = json.loads(line)
            qid = r.get("id")
            if "error" in r or "warning" in r:
                print("ENGINE MSG:", str(r)[:300])
                if "error" in r:
                    # KataGo answers a bad query with ONE error line: drop all of that query's still-missing turns at once
                    if qid not in remaining:
                        raise RuntimeError("engine error without a known query id: " + str(r)[:300])
                    self.last_errors[qid] = str(r.get("error"))[:200]
                    got += remaining[qid]; remaining[qid] = 0
                    continue
            if qid in remaining:
                if remaining[qid] == 0:
                    continue                      # late line of a query that already errored
                remaining[qid] -= 1
            res.append(r); got += 1
        th.join()
        return res, time.time() - t0

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(30)
        except Exception:
            self.p.kill()


def mk_query(qid, moves, turns, visits, komi=6.5, rules="japanese", profile=None, policy=False, nsym=1,
             initial_stones=None, initial_player=None):
    q = {"id": qid, "moves": moves_gtp(moves), "rules": rules, "komi": komi, "boardXSize": 19,
         "boardYSize": 19, "analyzeTurns": list(turns), "maxVisits": visits, "includePolicy": policy}
    if profile:
        q["overrideSettings"] = {"humanSLProfile": profile, "rootNumSymmetriesToSample": nsym}
    if initial_stones:      # handicap games only (HD>=2); current-format queries stay byte-identical
        q["initialStones"] = stones_gtp(initial_stones)
        if initial_player:
            q["initialPlayer"] = initial_player
    return q


def logp_of(policy, rc):
    p = policy[move_idx(rc)]
    return math.log(max(p, 1e-9))
