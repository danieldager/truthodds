"""RateGate arithmetic, with an injected clock -- no sleeping, no network."""
from deepinfra_runner import RateGate


def _clock():
    t = [1000.0]
    return t, (lambda: t[0])


def test_defaults_match_reader_lab():
    g = RateGate()
    assert (g.target, g.ceiling, g.floor) == (160.0, 200.0, 32.0)
    assert (g.window_s, g.breach_rate, g.cut_factor, g.recover_factor, g.recover_s, g.min_sample) \
        == (60.0, 0.05, 0.8, 0.10, 30.0, 20)


def test_single_429_never_cuts():
    t, c = _clock()
    g = RateGate(clock=c, log=lambda *a, **k: None)
    g.record(True)
    for _ in range(30):
        g.record(False)
    assert g.n_cuts == 0 and g.target == 160.0     # 1/31 = 3.2% < 5%


def test_windowed_breach_cuts_20pct():
    t, c = _clock()
    g = RateGate(clock=c, log=lambda *a, **k: None)
    for i in range(20):
        g.record(i < 3)                            # 15% over 20 -> breach
    assert g.n_cuts == 1
    assert g.target == 128.0                       # 160 * 0.8
    assert not g.events                            # window reset after acting


def test_cut_stops_at_floor():
    t, c = _clock()
    g = RateGate(clock=c, log=lambda *a, **k: None)
    for _ in range(40):
        for i in range(20):
            g.record(i < 5)
        t[0] += 31.0                     # cuts are rate-limited to one per half-window
    assert g.target == 32.0


def test_at_most_one_cut_per_half_window():
    t, c = _clock()
    g = RateGate(clock=c, log=lambda *a, **k: None)
    for _ in range(10):                  # a sustained storm within one half-window
        for i in range(20):
            g.record(i < 5)
    assert g.n_cuts == 1 and g.target == 128.0
    t[0] += 31.0
    for i in range(20):
        g.record(i < 5)
    assert g.n_cuts == 2


def test_thin_window_with_one_429_does_not_block_recovery():
    t, c = _clock()
    g = RateGate(clock=c, log=lambda *a, **k: None)
    g.target = 32.0
    t[0] += 40.0
    for i in range(10):                  # 10 completions, one 429: 10 % but below min_sample
        g.record(i == 0)
    assert g.target > 32.0


def test_recovers_after_clean_30s():
    t, c = _clock()
    g = RateGate(target=100, clock=c, log=lambda *a, **k: None)
    t[0] += 31
    g.record(False)
    assert g.n_ups == 1 and abs(g.target - 110.0) < 1e-9
    t[0] += 31
    g.record(False)
    assert abs(g.target - 121.0) < 1e-9


def test_inflight_is_capped_by_target():
    g = RateGate(target=2, log=lambda *a, **k: None)
    with g:
        with g:
            assert g.inflight == 2
    assert g.inflight == 0 and g.peak == 2
