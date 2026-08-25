"""The tempo/time-signature map's own tick <-> time conversions and its
resequencing after an edit (Phase 5: the workstation can now author this map,
not just display one read from a `.mid`).

`resequence` is the load-bearing function here: everything downstream of an
edit -- the ruler's bar lines, `Session._current_timing`, a `.mid` export --
reads `time_ms` off each marker rather than recomputing it, so a marker whose
`time_ms` disagrees with its own tick is a wrong answer nothing else can
detect on its own.
"""

from __future__ import annotations

import pytest

from snapmap_midi.music import timing as timing_module


def _timing(ticks_per_beat=480, tempo_changes=None, time_signatures=None):
    default_signature = {"tick": 0, "time_ms": 0.0, "numerator": 4, "denominator": 4}
    return {
        "ticks_per_beat": ticks_per_beat,
        "duration_ticks": 0,
        "grid_duration_ticks": 0,
        "tempo_changes": tempo_changes or [{"tick": 0, "time_ms": 0.0, "tempo": 500_000}],
        "time_signatures": time_signatures or [default_signature],
        "base_bpm": 120.0,
    }


def test_time_at_tick_and_tick_at_time_are_inverses_within_one_tempo_segment():
    timing = _timing()
    for tick in (0, 240, 480, 1920, 4800):
        ms = timing_module.time_at_tick(timing, tick)
        assert timing_module.tick_at_time(timing, ms) == pytest.approx(tick, abs=1)


def test_resequence_recomputes_time_ms_from_tick_positions_alone():
    """A marker's tick is the fact an edit changes; its `time_ms` has to
    follow, not be carried over stale from before the edit."""
    timing = _timing(
        tempo_changes=[
            {"tick": 0, "time_ms": 999.0, "tempo": 500_000},  # deliberately wrong
            {"tick": 960, "time_ms": 111.0, "tempo": 500_000},  # deliberately wrong
        ]
    )
    timing_module.resequence(timing)
    changes = timing["tempo_changes"]
    assert changes[0] == {"tick": 0, "time_ms": 0.0, "tempo": 500_000}
    # 960 ticks at 500,000 us/beat and 480 ticks/beat is exactly one second.
    assert changes[1]["time_ms"] == pytest.approx(1000.0)


def test_resequence_moves_a_time_signatures_time_ms_when_an_earlier_tempo_changes():
    """The core claim this whole feature depends on: a time signature never
    sets its own clock -- it only reads the tempo map's. Changing the tempo
    BEFORE a signature point must move that signature's own `time_ms`, with
    nobody having touched the signature itself.
    """
    timing = _timing(
        tempo_changes=[{"tick": 0, "time_ms": 0.0, "tempo": 500_000}],
        time_signatures=[
            {"tick": 0, "time_ms": 0.0, "numerator": 4, "denominator": 4},
            {"tick": 1920, "time_ms": 2000.0, "numerator": 3, "denominator": 4},
        ],
    )
    timing_module.resequence(timing)
    before = timing["time_signatures"][1]["time_ms"]
    assert before == pytest.approx(2000.0)

    # Double the tempo from the very start.
    timing["tempo_changes"][0]["tempo"] = 250_000
    timing_module.resequence(timing)
    after = timing["time_signatures"][1]["time_ms"]
    assert after == pytest.approx(before / 2)


def test_resequence_recomputes_base_bpm_from_the_first_tempo_marker():
    timing = _timing()
    timing["tempo_changes"][0]["tempo"] = 400_000
    timing_module.resequence(timing)
    assert timing["base_bpm"] == pytest.approx(round(60_000_000.0 / 400_000, 2))


def test_resequence_recomputes_source_and_grid_duration_ms_from_ticks():
    timing = _timing()
    timing["duration_ticks"] = 960
    timing["grid_duration_ticks"] = 1920
    timing_module.resequence(timing)
    assert timing["source_duration_ms"] == pytest.approx(1000.0)
    assert timing["grid_duration_ms"] == pytest.approx(2000.0)

    # Halving the tempo doubles how long the same tick positions take.
    timing["tempo_changes"][0]["tempo"] = 1_000_000
    timing_module.resequence(timing)
    assert timing["source_duration_ms"] == pytest.approx(2000.0)
    assert timing["grid_duration_ms"] == pytest.approx(4000.0)


def test_resequence_sorts_markers_by_tick_regardless_of_input_order():
    timing = _timing(
        tempo_changes=[
            {"tick": 1920, "time_ms": 0.0, "tempo": 500_000},
            {"tick": 0, "time_ms": 0.0, "tempo": 500_000},
            {"tick": 960, "time_ms": 0.0, "tempo": 250_000},
        ]
    )
    timing_module.resequence(timing)
    ticks = [marker["tick"] for marker in timing["tempo_changes"]]
    assert ticks == sorted(ticks)
    assert ticks[0] == 0


def test_resequence_recovers_from_a_map_with_no_tick_zero_entry():
    """A defensive fallback, not a case any caller here is expected to
    produce -- every add/edit method keeps a tick-0 entry -- but a malformed
    map should resequence to something playable rather than raise."""
    timing = _timing(tempo_changes=[{"tick": 480, "time_ms": 500.0, "tempo": 500_000}])
    timing_module.resequence(timing)
    assert timing["tempo_changes"][0]["tick"] == 0
