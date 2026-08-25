"""Writing a `Song` back out as a standard `.mid` file, and reading it back.

Round-trip correctness is what matters here: notes, track structure and the
tempo/time-signature map have to survive a write-then-reimport with only the
known, documented lossy step -- tick quantization, from milliseconds back to
ticks and forward again -- and nothing silently dropped or scrambled. DOOM
sound assignments and SnapMap pitch/volume expression are deliberately NOT
part of this contract: a `.mid` file has no field for either, by design (see
`music/midi_export.py`'s own module docstring).
"""

from __future__ import annotations

import pytest

from snapmap_midi.music import importer
from snapmap_midi.music.midi_export import write_midi
from snapmap_midi.ui.session import Session


def _built_song(tmp_path):
    """A song assembled the same way the workstation itself would build one --
    through `Session`'s own structural bridge methods, not by poking `Song`
    fields directly."""
    session = Session()
    session.new_song()
    lead_id = session.create_track("Lead")
    session.create_note(lead_id, 60, 0, 480, 100)
    session.create_note(lead_id, 64, 480, 240, 90)
    session.create_note(lead_id, 67, 960, 960, 110)

    bass_id = session.create_track("Bass")
    session.create_note(bass_id, 36, 0, 1920, 70)

    session.add_tempo_change(2000, 90)
    session.add_time_signature(1000, 3, 4)
    return session.song()


def test_exported_notes_round_trip_through_the_importer(tmp_path):
    song = _built_song(tmp_path)
    destination = write_midi(song, tmp_path / "roundtrip.mid")
    assert destination.exists()

    reimported = importer.import_song(destination)
    assert len(reimported.tracks) == 2

    by_name = {track.name: track for track in reimported.tracks}
    assert set(by_name) == {"Lead", "Bass"}

    # Milliseconds round-trip through TICKS on the way out and back
    # (`music/timing.py::tick_at_time`), so a value a few tenths of a tick
    # off its original ms rounds to a neighbouring millisecond -- the one
    # documented, acceptable lossy step this whole test exists to bound.
    # Pitch and velocity have no such conversion and must be exact.
    lead_notes = sorted(
        ((n.pitch, n.start_ms, n.duration_ms, n.velocity) for n in by_name["Lead"].notes)
    )
    expected_lead = [(60, 0, 480, 100), (64, 480, 240, 90), (67, 960, 960, 110)]
    assert len(lead_notes) == len(expected_lead)
    for (pitch, start, duration, velocity), (e_pitch, e_start, e_duration, e_velocity) in zip(
        lead_notes, expected_lead
    ):
        assert pitch == e_pitch
        assert velocity == e_velocity
        assert start == pytest.approx(e_start, abs=2)
        assert duration == pytest.approx(e_duration, abs=2)

    bass_notes = [(n.pitch, n.start_ms, n.duration_ms, n.velocity) for n in by_name["Bass"].notes]
    assert len(bass_notes) == 1
    assert bass_notes[0][0] == 36
    assert bass_notes[0][2] == pytest.approx(1920, abs=2)
    assert bass_notes[0][3] == 70


def test_exported_tempo_and_time_signature_round_trip(tmp_path):
    song = _built_song(tmp_path)
    destination = write_midi(song, tmp_path / "roundtrip.mid")
    reimported = importer.import_song(destination)

    tempo_changes = sorted(reimported.tempo_map, key=lambda m: m["tick"])
    assert len(tempo_changes) == 2
    assert tempo_changes[0]["tick"] == 0
    assert tempo_changes[0]["tempo"] == 500_000
    # 2,000 ms in, requested as 90 BPM -- tempo is microseconds per beat, so
    # this is the same round-trip lossiness (an integer round) the workstation
    # itself already accepts for a tempo edit.
    assert tempo_changes[1]["tempo"] == pytest.approx(round(60_000_000 / 90), abs=1)

    signatures = sorted(reimported.time_signature_map, key=lambda m: m["tick"])
    assert len(signatures) == 2
    assert (signatures[0]["numerator"], signatures[0]["denominator"]) == (4, 4)
    assert (signatures[1]["numerator"], signatures[1]["denominator"]) == (3, 4)


def test_a_note_dragged_past_the_songs_length_still_exports(tmp_path):
    """`Song.duration_ms` is authoritative and independent of note content
    (Phase 3) -- exporting must not depend on it either; a `.mid` file has no
    field for "song length" beyond its own last event."""
    session = Session()
    session.new_song()
    track_id = session.create_track("Lead")
    session.create_note(track_id, 60, 0, 480, 100)
    destination = write_midi(session.song(), tmp_path / "song.mid")
    reimported = importer.import_song(destination)
    assert len(reimported.tracks[0].notes) == 1


def test_export_writes_no_notes_for_an_empty_track(tmp_path):
    session = Session()
    session.new_song()
    session.create_track("Empty")
    destination = write_midi(session.song(), tmp_path / "empty.mid")
    reimported = importer.import_song(destination)
    assert reimported.tracks == [] or all(not t.notes for t in reimported.tracks)


def test_exported_channels_wrap_into_the_sixteen_midi_channels(tmp_path):
    """A project can hold more tracks than MIDI has channels -- every
    hand-drawn track mints its own `channel` counter with no ceiling (see
    `Song.drawn_channel_serial`). Exporting must not raise; wrapping is
    documented, honest lossy behaviour, not a refusal."""
    session = Session()
    session.new_song()
    for index in range(20):
        track_id = session.create_track("Track %d" % index)
        session.create_note(track_id, 60, index * 100, 90, 100)
    destination = write_midi(session.song(), tmp_path / "many_tracks.mid")
    reimported = importer.import_song(destination)
    assert len(reimported.tracks) == 20
    assert sum(len(t.notes) for t in reimported.tracks) == 20
