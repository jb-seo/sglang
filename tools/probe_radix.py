#!/usr/bin/env python3
"""Where does the prefix cache actually hit, and where does it stop?

Runs against a LIVE SGLang server. No restart, no config change.

Two modes:

  ab      Two sessions sharing no prefix, at a SHORT and a LONG context, with
          one other-session request in between. Isolates one variable: does an
          intervening session kill the cache, and does that depend on length?

              python3 probe_radix.py --mode ab

  simulate  Several sessions that start short and grow a turn at a time, called
          in random order and optionally overlapping -- the shape of real
          multi-window use, where every session's history accumulates while
          the others push it down the LRU. Each turn carries the previous
          reply, so the prefix grows the way a chat's does. Reports, per call,
          how much of the prefix that SHOULD have been cached actually was.

              python3 probe_radix.py --mode simulate --turns 40 \
                  --concurrency 3 --gen-tokens 256

`ab` answers a yes/no question; `simulate` shows where the hit rate falls off
as contexts grow and sessions interleave, which is the failure people actually
hit. Start with `ab`, use `simulate` to find the knee.

Watch the server log alongside:
    grep -E "Prefill batch|mamba evictable" <serverlog> | tail -40
"""
import argparse, json, random, sys, threading, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def gen(url, prompt, max_new_tokens=1, timeout=900):
    """One call. Returns (secs, prompt_tokens, cached_tokens, completion_tokens,
    text).

    max_new_tokens matters more than it looks. A request holds its mamba slots
    for as long as it runs, and crosses a state-tracking boundary every
    mamba_track_interval decoded tokens -- so a 1-token call measures prefill
    reuse while barely occupying the server, which is not the workload anyone
    actually runs.
    """
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(
            {"text": prompt,
             "sampling_params": {"max_new_tokens": max_new_tokens,
                                 "temperature": 0.0}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    text = out.get("text", "") if isinstance(out, dict) else ""
    return (dt, meta.get("prompt_tokens"), meta.get("cached_tokens"),
            meta.get("completion_tokens"), text)


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
            dt, ptok, ctok, _, _ = gen(url, prompt)
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
    """One conversation, with the three things real ones do to their history.

    grow     append a turn. The previous prompt and reply stay a strict
             prefix, so the tree should hand all of it back.
    shrink    drop the oldest turns to stay under a limit. The prefix changes
             at token 0, so a miss here is correct, not a failure.
    compact   replace the history with a summary -- a new system prompt and a
             condensed transcript. Also a miss by construction, and the
             expensive one, since what follows is a full prefill.

    Separating them matters: lumping compactions in with growth reads as a
    broken cache, and hiding them reads as a workload nobody runs.
    """

    def __init__(self, tag, rng, grow_min, grow_max):
        self.tag = tag
        self.rng = rng
        self.grow_min, self.grow_max = grow_min, grow_max
        self.words: list[str] = []
        self.turn = 0
        self.epoch = 0          # bumped on compaction: changes the prefix
        self.last_reply = ""
        self.prev_cacheable_tokens: int | None = None

    def est_tokens(self) -> int:
        return int(len(self.words) * 1.3)

    def _tok(self, i: int) -> str:
        return f"{self.tag}e{self.epoch}w{i}"

    def grow(self) -> str:
        n = self.rng.randint(self.grow_min, self.grow_max)
        base = len(self.words)
        if self.last_reply:
            self.words += self.last_reply.split()
            self.last_reply = ""
        self.words += [self._tok(base + i) for i in range(n)]
        return "grow"

    def shrink(self) -> str:
        """Drop the oldest third: the head changes, so nothing matches."""
        self.last_reply = ""
        self.words = self.words[len(self.words) // 3 :]
        self.epoch += 1
        self.words = [self._tok(i) for i in range(len(self.words))]
        return "shrink"

    def compact(self) -> str:
        """Summarise: a fresh, much shorter history under a new prefix."""
        self.last_reply = ""
        keep = max(self.grow_min, len(self.words) // 8)
        self.epoch += 1
        self.words = [self._tok(i) for i in range(keep)]
        return "compact"

    def next_prompt(self, action: str) -> str:
        act = {"grow": self.grow, "shrink": self.shrink,
               "compact": self.compact}[action]()
        self.turn += 1
        return " ".join(self.words) + f" turn{self.turn}?"


def run_simulate(url, args):
    rng = random.Random(args.seed)
    tags = [chr(ord("A") + i) for i in range(args.sessions)]
    sessions = {
        t: Session(t, random.Random(args.seed + i), args.grow_min, args.grow_max)
        for i, t in enumerate(tags)
    }
    # Fix the call ORDER from the seed, before any timing is involved, so two
    # runs at different concurrency compare the same workload. Only the
    # overlap between calls varies.
    order = [rng.choice(tags) for _ in range(args.turns)]
    # Actions are drawn here too, before any timing, so runs stay comparable.
    # The budget check is the exception: it depends on how big sessions have
    # actually grown, so it is applied at issue time and can override.
    draws = [rng.random() for _ in range(args.turns)]

    print(f"\n{args.sessions} sessions, {args.turns} calls, "
          f"concurrency={args.concurrency}, gen_tokens={args.gen_tokens}, "
          f"seed={args.seed}")
    print("'reuse' is cached tokens as a fraction of the previous turn's prompt")
    print("-- the part the tree already held. 100% = perfect, 0% = full recompute.")
    print("'inflt' is how many calls were in flight when this one was issued.\n")
    print(f"{'#':>3} {'sess':>4} {'turn':>4} {'action':>8} {'inflt':>5} "
          f"{'prompt':>8} {'cached':>8} {'expect':>8} {'reuse':>8} {'secs':>7}")
    print("(* = shrink/compact rewrote the prefix; a miss there is correct)")

    lock = threading.Lock()
    inflight: set[str] = set()
    peak = 0
    rows = []

    def do_call(seq, sess, expect, prompt, at_issue, action):
        nonlocal rows
        try:
            dt, ptok, ctok, gtok, text = gen(url, prompt, args.gen_tokens)
        except Exception as e:
            with lock:
                inflight.discard(sess.tag)
            print(f"{seq:>3} {sess.tag:>4} {sess.turn:>4}  FAILED: {e}",
                  file=sys.stderr)
            return
        reuse = None if not expect else (ctok or 0) / expect
        with lock:
            sess.last_reply = text
            sess.prev_cacheable_tokens = (ptok or 0) + (gtok or 0)
            inflight.discard(sess.tag)
            # A miss after shrink/compact is the correct answer, not a
            # failure: those rewrite the prefix on purpose.
            expected_miss = action in ("shrink", "compact")
            shown = "-" if reuse is None else (
                f"{reuse:6.1%}" + ("*" if expected_miss else "")
            )
            print(
                f"{seq:>3} {sess.tag:>4} {sess.turn:>4} {action:>8} {at_issue:>5} "
                f"{ptok:>8} {ctok:>8} {'-' if expect is None else expect:>8} "
                f"{shown:>8} {dt:>7.2f}"
            )
            if reuse is not None:
                rows.append((sess.tag, expect, reuse, at_issue, action))

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        futures = []
        for seq, tag in enumerate(order, 1):
            # A client keeps at most one request per window open, and turn N+1
            # is only meaningful after turn N: wait out this session, but let
            # the others keep running.
            while True:
                with lock:
                    if tag not in inflight and len(inflight) < args.concurrency:
                        sess = sessions[tag]
                        total = sum(x.est_tokens() for x in sessions.values())
                        r = draws[seq - 1]
                        if args.budget_tokens and total > args.budget_tokens:
                            # Over the space the server has: the biggest
                            # session compacts, which is what a client does
                            # when it runs out of window.
                            biggest = max(sessions.values(),
                                          key=lambda x: x.est_tokens())
                            if biggest.tag == tag:
                                action = "compact"
                            elif r < args.shrink_prob:
                                action = "shrink"
                            else:
                                action = "grow"
                        elif r < args.compact_prob:
                            action = "compact"
                        elif r < args.compact_prob + args.shrink_prob:
                            action = "shrink"
                        else:
                            action = "grow"
                        expect = sess.prev_cacheable_tokens
                        prompt = sess.next_prompt(action)
                        inflight.add(tag)
                        at_issue = len(inflight)
                        peak = max(peak, at_issue)
                        break
                time.sleep(0.005)
            futures.append(
                pool.submit(do_call, seq, sess, expect, prompt, at_issue, action))
        for f in futures:
            f.result()

    if not rows:
        print("\nno reusable turns recorded")
        return

    grow = [(t, e, r, c) for t, e, r, c, a in rows if a == "grow"]
    rewrote = [(t, e, r, c) for t, e, r, c, a in rows if a != "grow"]

    print("\n" + "=" * 72)
    print(f"peak concurrency {peak}   grow turns {len(grow)}   "
          f"prefix-rewriting turns {len(rewrote)}")
    print("Only grow turns are a verdict on the cache. The rest are counted "
          "so the\nworkload stays honest, not so the cache gets blamed for "
          "them.")
    print("=" * 72)

    if not grow:
        print("  no grow turns -- raise --turns or lower the rewrite probs")
        return

    print("\nreuse on grow turns, by context size")
    for lo, hi in [(0, 4000), (4000, 16000), (16000, 64000),
                   (64000, 200000), (200000, 10**9)]:
        got = [r for _, e, r, _ in grow if lo <= e < hi]
        if not got:
            continue
        hi_s = "inf" if hi > 10**8 else str(hi)
        print(f"  {lo:>7}-{hi_s:<7} n={len(got):<4} "
              f"mean={sum(got)/len(got):6.1%}  min={min(got):6.1%}  "
              f"wipeouts={sum(1 for r in got if r < 0.01)}")

    print("\nreuse on grow turns, by calls in flight")
    by_c = defaultdict(list)
    for _, _, r, c in grow:
        by_c[c].append(r)
    for c in sorted(by_c):
        got = by_c[c]
        print(f"  {c} in flight  n={len(got):<4} mean={sum(got)/len(got):6.1%}  "
              f"min={min(got):6.1%}  wipeouts={sum(1 for r in got if r < 0.01)}")

    per = defaultdict(list)
    for tag, _, r, _ in grow:
        per[tag].append(r)
    print("\nreuse on grow turns, by session")
    for tag in sorted(per):
        got = per[tag]
        print(f"  {tag}  n={len(got):<4} mean={sum(got)/len(got):6.1%}  "
              f"min={min(got):6.1%}  wipeouts={sum(1 for r in got if r < 0.01)}")

    vals = [r for _, _, r, _ in grow]
    wipe = sum(1 for r in vals if r < 0.01)
    partial = [r for r in vals if 0.01 <= r < 0.995]
    print(f"\ngrow-turn mean reuse: {sum(vals)/len(vals):.1%}")
    print(f"  wipeouts (0%): {wipe}/{len(vals)}")
    if partial:
        lost = [int(e * (1 - r)) for _, e, r, _ in grow if 0.01 <= r < 0.995]
        print(f"  partial:       {len(partial)}/{len(vals)}, "
              f"losing {min(lost)}-{max(lost)} tokens off the tail")
    print("""
  A wipeout and a partial are different failures.

    partial, a few hundred tokens, roughly constant  -> the match stops at the
      last state checkpoint. Bounded by mamba_track_interval; the fraction
      shrinks as the context grows. Not a leak.
    wipeout, the whole prefix                        -> that session's
      checkpoints are gone. Look at which sessions grew while it sat idle, and
      at `mamba evictable` in the server log across the same window.

  Run the same seed at --concurrency 1 and higher, and with --budget-tokens on
  and off, changing one at a time. Same order, same prompts, same actions.
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
    ap.add_argument("--concurrency", type=int, default=1,
                    help="simulate: max calls in flight (1 = strictly serial)")
    ap.add_argument("--budget-tokens", type=int, default=0,
                    help="simulate: total token space across sessions. When "
                         "exceeded the largest session compacts, the way a "
                         "client does when it runs out of window. 0 = off.")
    ap.add_argument("--compact-prob", type=float, default=0.0,
                    help="simulate: chance a turn compacts instead of growing")
    ap.add_argument("--shrink-prob", type=float, default=0.0,
                    help="simulate: chance a turn drops its oldest third")
    ap.add_argument("--gen-tokens", type=int, default=256,
                    help="simulate: tokens each call generates. 1 measures "
                         "prefill reuse only; a request holds its mamba slots "
                         "for as long as it runs, so keep this realistic.")
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
