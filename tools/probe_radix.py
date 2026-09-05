#!/usr/bin/env python3
"""Where does the prefix cache actually hit, and where does it stop?

Runs against a LIVE SGLang server. No restart, no config change.

Two modes:

  ab      Two sessions sharing no prefix, at a SHORT and a LONG context, with
          one other-session request in between. Isolates one variable: does an
          intervening session kill the cache, and does that depend on length?

              python3 probe_radix.py --mode ab

  simulate  Several sessions that start short and grow a turn at a time, called
          in random order -- the shape of real multi-window use, where every
          session's history accumulates while the others push it down the LRU.
          Reports, per call, how much of the prefix that SHOULD have been
          cached actually was.

              python3 probe_radix.py --mode simulate --turns 40

`ab` answers a yes/no question; `simulate` shows where the hit rate falls off
as contexts grow and sessions interleave, which is the failure people actually
hit. Start with `ab`, use `simulate` to find the knee.

Watch the server log alongside:
    grep -E "Prefill batch|mamba evictable" <serverlog> | tail -40
"""
import argparse, json, random, sys, time, urllib.request
from collections import defaultdict


def gen(url, prompt, timeout=900):
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(
            {"text": prompt,
             "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    return dt, meta.get("prompt_tokens"), meta.get("cached_tokens")


def filler(tag, n_words):
    """Deterministic and distinct per tag: same tag => identical prefix."""
    return " ".join(f"{tag}{i}" for i in range(n_words))


def run_round(url, words, label, other_words=None, tag=""):
    """One A/B/A probe. `other_words` sizes the intervening session separately,
    so 'A is long' and 'the request that interrupted it was long' are not the
    same knob -- in a single-size round they move together and the result
    cannot say which one mattered."""
    other_words = words if other_words is None else other_words
    A = filler("alpha" + tag, words)
    B = filler("bravo" + tag, other_words)   # shares nothing with A
    steps = [
        ("A1 cold A",                 A + " Q1"),
        ("A2 A again, no gap",        A + " Q2"),
        ("B1 other session",          B + " Q1"),
        ("A3 back to A  <-- KEY",     A + " Q3"),
    ]
    print(f"\n=== {label}: A~{words}w, B~{other_words}w ===")
    rows = []
    for name, prompt in steps:
        try:
            dt, ptok, ctok = gen(url, prompt)
        except Exception as e:
            print(f"  {name:24s} FAILED: {e}", file=sys.stderr)
            rows.append((name, None, None, None))
            continue
        pct = (100.0 * ctok / ptok) if (ptok and ctok is not None) else None
        rows.append((name, dt, ptok, ctok))
        print(f"  {name:24s} {dt:7.2f}s  prompt={ptok:<7} cached={ctok:<7}"
              f" {'' if pct is None else f'({pct:.1f}%)'}")
    return rows


def verdict(rows):
    """A2 = same session repeat, A3 = repeat after an intervening session."""
    by = {r[0].split()[0]: r for r in rows}
    a2, a3 = by.get("A2"), by.get("A3")
    if not a2 or not a3 or a2[2] is None or a3[2] is None:
        return "incomplete"
    a2_hit = (a2[3] or 0) > 0.5 * (a2[2] or 1)
    a3_hit = (a3[3] or 0) > 0.5 * (a3[2] or 1)
    if a2_hit and a3_hit:
        return "survives alternation"
    if a2_hit and not a3_hit:
        return "KILLED by the other session"
    return "no hit even without a gap"



class Session:
    """One conversation that grows a turn at a time.

    Each turn appends to the previous prompt, so turn N's prompt has turn
    N-1's whole prompt as a strict prefix -- exactly what the radix tree is
    supposed to hand back. That prior length is the yardstick: hitting 90% of
    the total prompt means nothing if the last 10% was all that was new.
    """

    def __init__(self, tag: str, rng: random.Random, grow_min: int, grow_max: int):
        self.tag = tag
        self.rng = rng
        self.grow_min, self.grow_max = grow_min, grow_max
        self.words: list[str] = []
        self.turn = 0
        self.prev_prompt_tokens: int | None = None

    def next_prompt(self) -> str:
        """Append a turn's worth of words and return the whole conversation."""
        n = self.rng.randint(self.grow_min, self.grow_max)
        base = len(self.words)
        self.words += [f"{self.tag}{base + i}" for i in range(n)]
        self.turn += 1
        return " ".join(self.words) + f" turn{self.turn}?"


def run_simulate(url, args):
    rng = random.Random(args.seed)
    tags = [chr(ord("A") + i) for i in range(args.sessions)]
    sessions = {
        t: Session(t, random.Random(args.seed + i), args.grow_min, args.grow_max)
        for i, t in enumerate(tags)
    }

    print(f"\n{args.sessions} sessions, {args.turns} calls, seed={args.seed}")
    print("'reuse' is cached tokens as a fraction of the previous turn's prompt")
    print("-- the part the tree already held. 100% = perfect, 0% = full recompute.\n")
    print(f"{'#':>3} {'sess':>4} {'turn':>4} {'prompt':>8} {'cached':>8} "
          f"{'expect':>8} {'reuse':>7} {'secs':>7}")

    rows = []
    for i in range(1, args.turns + 1):
        sess = sessions[rng.choice(tags)]
        expect = sess.prev_prompt_tokens
        prompt = sess.next_prompt()
        try:
            dt, ptok, ctok = gen(url, prompt)
        except Exception as e:
            print(f"{i:>3} {sess.tag:>4} {sess.turn:>4}  FAILED: {e}", file=sys.stderr)
            continue

        reuse = None if not expect else (ctok or 0) / expect
        print(
            f"{i:>3} {sess.tag:>4} {sess.turn:>4} {ptok:>8} {ctok:>8} "
            f"{'-' if expect is None else expect:>8} "
            f"{'-' if reuse is None else f'{reuse:6.1%}':>7} {dt:>7.2f}"
        )
        if reuse is not None:
            rows.append((sess.tag, expect, reuse))
        sess.prev_prompt_tokens = ptok

    if not rows:
        print("\nno reusable turns recorded")
        return

    print("\n" + "=" * 62)
    print("reuse by context size (the previous turn's prompt)")
    print("=" * 62)
    buckets = [(0, 4000), (4000, 16000), (16000, 64000), (64000, 10**9)]
    for lo, hi in buckets:
        got = [r for _, e, r in rows if lo <= e < hi]
        if not got:
            continue
        hi_s = "inf" if hi > 10**8 else str(hi)
        print(f"  {lo:>6}-{hi_s:<6} n={len(got):<4} "
              f"mean={sum(got)/len(got):6.1%}  min={min(got):6.1%}")

    per = defaultdict(list)
    for tag, _, r in rows:
        per[tag].append(r)
    print("\nreuse by session")
    for tag in sorted(per):
        got = per[tag]
        print(f"  {tag}  n={len(got):<4} mean={sum(got)/len(got):6.1%}  min={min(got):6.1%}")

    overall = sum(r for _, _, r in rows) / len(rows)
    print(f"\noverall mean reuse: {overall:.1%}")
    print("""
  near 100% throughout      -> the cache is doing its job; look elsewhere.
  falls off as size grows   -> length-dependent eviction. Compare the knee
                               against --max-mamba-cache-size and the
                               chunks-per-prefill count.
  collapses to ~0 for some  -> those turns lost their whole prefix, not part
                               of it. That is a tombstoned/absent checkpoint,
                               not gradual pressure.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8010")
    ap.add_argument("--mode", choices=["ab", "simulate"], default="ab")
    ap.add_argument("--short", type=int, default=1500, help="ab: words, short round")
    ap.add_argument("--long", type=int, default=20000, help="ab: words, long round")
    ap.add_argument("--sessions", type=int, default=3, help="simulate: how many")
    ap.add_argument("--turns", type=int, default=40, help="simulate: total calls")
    ap.add_argument("--grow-min", type=int, default=200,
                    help="simulate: fewest words a turn adds")
    ap.add_argument("--grow-max", type=int, default=2500,
                    help="simulate: most words a turn adds")
    ap.add_argument("--seed", type=int, default=0, help="simulate: reproducible order")
    ap.add_argument("--repeat", type=int, default=3,
                    help="ab: repeats per cell (1 tells you nothing)")
    a = ap.parse_args()

    if a.mode == "simulate":
        run_simulate(a.url, a)
        return

    # Four cells, so a miss can be attributed. If only (long A, long B) fails
    # the pressure needs both; if every long-B cell fails it is the
    # interrupting request's size; if every long-A cell fails it is the
    # probed session's own.
    cells = [
        ("short A / short B", a.short, a.short),
        ("short A / LONG  B", a.short, a.long),
        ("LONG  A / short B", a.long, a.short),
        ("LONG  A / LONG  B", a.long, a.long),
    ]
    results = {name: [] for name, _, _ in cells}
    for rep in range(a.repeat):
        for ci, (name, w, ow) in enumerate(cells):
            # A fresh tag per (repeat, cell). Sharing one across cells lets a
            # cell hit on the previous cell's tree, which measures nothing.
            rows = run_round(a.url, w, f"[rep {rep + 1}] {name}",
                             other_words=ow, tag=f"r{rep}c{ci}")
            results[name].append(verdict(rows))

    print("\n" + "=" * 62)
    print(f"  {a.repeat} repeat(s) per cell -- a single run proves nothing here")
    print("=" * 62)
    for name, _, _ in cells:
        vs = results[name]
        agree = "consistent" if len(set(vs)) == 1 else "INCONSISTENT"
        print(f"  {name:20s} {agree:12s} {vs}")
    print("=" * 62)
    print("""
  Read the four cells together; one cell on its own attributes nothing.

    only LONG A / LONG B fails  -> pressure needs both sides large.
    both LONG-B cells fail      -> the INTERRUPTING request's size is what
                                   evicts; the probed session's length is
                                   incidental.
    both LONG-A cells fail      -> the probed session's own length decides;
                                   what interrupted it does not matter.
    all four fail               -> length is not the variable at all.
    none fail                   -> alternation is innocent here; the real
                                   trigger is elsewhere (Stop, compaction, a
                                   changed system prompt).

  Any cell marked INCONSISTENT means the run-to-run variance is larger than
  the effect. Raise --repeat before drawing anything from it.
""")


if __name__ == "__main__":
    main()
