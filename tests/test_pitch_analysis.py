"""Conservative root-pitch classification for decoded soundbank media."""

from __future__ import annotations

import math

import pytest

from snapmap_midi.audio import pitch
from snapmap_midi.audio.pitch import analyze_pcm, analyze_sources

_RATE = 8000


def _tone(frequency: float, seconds: float = 1.0) -> list[int]:
    return [
        round(12000 * math.sin(2 * math.pi * frequency * index / _RATE))
        for index in range(round(_RATE * seconds))
    ]


def _source(media_id: int, frequency: float) -> dict:
    return {
        "media_id": media_id,
        "rate": _RATE,
        "per_channel": [_tone(frequency)],
    }


def test_yin_resolves_a_concert_a_root():
    estimate, reason = analyze_pcm(_RATE, [_tone(440.0)])
    assert reason == "pitched"
    assert estimate is not None
    assert estimate.root_midi == pytest.approx(69.0, abs=0.15)
    assert estimate.confidence >= 0.75


def test_silence_is_never_given_a_plausible_root():
    estimate, reason = analyze_pcm(_RATE, [[0] * _RATE])
    assert estimate is None
    assert reason == "silence"


def test_agreeing_event_leaves_produce_one_numeric_profile():
    profile = analyze_sources([_source(1, 440.0), _source(2, 440.0)])
    assert profile["classification"] == "pitched"
    assert profile["pitchable"] is True
    assert profile["root_midi"] == pytest.approx(69.0, abs=0.15)
    assert profile["sources"] == 2


def test_random_container_leaves_with_different_roots_are_variable():
    profile = analyze_sources([_source(1, 440.0), _source(2, 523.251)])
    assert profile["classification"] == "variable"
    assert profile["pitchable"] is False
    assert profile["root_midi"] is None
    assert profile["reason"] == "event leaves have different roots"


def test_one_pitched_and_one_silent_leaf_is_not_accepted():
    profile = analyze_sources(
        [
            _source(1, 440.0),
            {"media_id": 2, "rate": _RATE, "per_channel": [[0] * _RATE]},
        ]
    )
    assert profile["classification"] == "variable"
    assert profile["pitchable"] is False
    assert profile["reason"] == "leaf classifications disagree"


def test_no_decodable_media_is_an_unavailable_profile():
    profile = analyze_sources([])
    assert profile == {
        "classification": "unavailable",
        "pitchable": False,
        "root_midi": None,
        "confidence": 0.0,
        "cents_spread": None,
        "sources": 0,
        "reason": "no decodable media",
    }


def test_a_high_partial_is_not_promoted_to_the_sound_root(monkeypatch):
    monkeypatch.setattr(pitch, "_yin", lambda frame, rate: (1000.0, 0.95))
    monkeypatch.setattr(pitch, "_dominant_frequency", lambda frame, rate: 500.0)

    estimate, reason = pitch.analyze_pcm(_RATE, [_tone(1000.0)])
    assert estimate is None
    assert reason == "harmonic_ambiguity"

    profile = pitch.analyze_sources([_source(1, 1000.0)])
    assert profile["classification"] == "ambiguous"
    assert profile["pitchable"] is False
    assert profile["root_midi"] is None
    assert profile["relative_recommended"] is True


def test_weak_but_consistent_windows_resolve_a_root(monkeypatch):
    # Real acoustic media (decay noise, mild clipping, a plucked string's
    # transient) rarely clears the strong per-frame bar even when clearly,
    # consistently periodic across the whole clip. Cross-window agreement --
    # several independent windows landing on the same pitch -- is trusted in
    # its place instead of demanding one frame be individually confident.
    monkeypatch.setattr(pitch, "_yin", lambda frame, rate: (110.0, 0.55))
    monkeypatch.setattr(pitch, "_dominant_frequency", lambda frame, rate: 220.0)

    estimate, reason = pitch.analyze_pcm(_RATE, [_tone(110.0)])
    assert reason == "pitched"
    assert estimate is not None
    assert estimate.root_midi == pytest.approx(pitch.frequency_to_midi(110.0), abs=0.05)


def test_a_subharmonic_candidate_is_refused_rather_than_guessed(monkeypatch):
    """YIN's classic failure is period doubling: it reports a note an octave
    BELOW the real one. The existing guard only catches it reaching UP to a
    high partial -- a sub-harmonic sits below the dominant component and sails
    through that check.

    A frame with no energy at the claimed root but plenty an octave up is
    equally consistent with a sub-harmonic error and with a genuine weak
    fundamental. Those readings are an octave apart and the frame cannot
    separate them, so the module's own rule applies: prefer "unknown" to a
    plausible wrong octave. Reported live as `Play_sfx_stranglerstrings_02`,
    where a 0.545-confidence root of D2 was accepted while the sample's
    strongest component sat an octave above it at D3."""
    rate = _RATE
    # Energy only at 220 Hz, while YIN claims the root is an octave below.
    frames = [_tone(220.0)]
    monkeypatch.setattr(pitch, "_yin", lambda frame, r: (110.0, 0.95))
    monkeypatch.setattr(pitch, "_dominant_frequency", lambda frame, r: 220.0)

    estimate, reason = pitch.analyze_pcm(rate, frames)
    assert estimate is None
    assert reason == "harmonic_ambiguity"

    profile = pitch.analyze_sources([_source(1, 220.0)])
    assert profile["classification"] == "ambiguous"
    assert profile["pitchable"] is False
    assert profile["relative_recommended"] is True


def test_a_supported_fundamental_still_resolves(monkeypatch):
    """The guard must only fire on a genuinely unsupported root. A tone with
    real energy at the frequency YIN reports is exactly the case it must not
    touch."""
    monkeypatch.setattr(pitch, "_dominant_frequency", lambda frame, r: 220.0)

    estimate, reason = pitch.analyze_pcm(_RATE, [_tone(220.0)])
    assert reason == "pitched"
    assert estimate is not None
    assert estimate.root_midi == pytest.approx(pitch.frequency_to_midi(220.0), abs=0.2)


def test_a_single_weak_window_is_not_trusted_alone(monkeypatch):
    # One window clearing only the coalesced bar is indistinguishable from a
    # lucky-looking dip in noise. Unlike a single STRONG window, it must not
    # stand alone -- it needs another window to agree with it first.
    calls = {"count": 0}

    def flaky_yin(frame, rate):
        calls["count"] += 1
        return (110.0, 0.55) if calls["count"] == 1 else None

    monkeypatch.setattr(pitch, "_yin", flaky_yin)
    monkeypatch.setattr(pitch, "_dominant_frequency", lambda frame, rate: 220.0)

    estimate, reason = pitch.analyze_pcm(_RATE, [_tone(110.0)])
    assert estimate is None
    assert reason == "unstable"
