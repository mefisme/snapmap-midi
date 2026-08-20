"""The editable song: importing one, keeping one, and not changing what it means.

The refactor this covers replaced "re-read the `.mid` on every compile" with
"hold a song and edit it". The whole risk of that change is silent: a song that
converts almost the same as the file it came from produces a map that loads,
plays, and is wrong in a way nobody can point at. So the load-bearing assertion
here is byte identity -- the same file with the same settings has to compile to
the same bytes through the song as it did through the parser -- and everything
else is about the things a song can now do that a file could not.
"""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from snapmap_midi import project
from snapmap_midi import settings as settings_module
from snapmap_midi.compile import compile_song, compile_to_rawmap
from snapmap_midi.music import importer
from snapmap_midi.music.levers import compile_levers
from snapmap_midi.music.midi import for_part
from snapmap_midi.music.song import SONG_VERSION, SongError, from_dict, loop_window, to_dict
from snapmap_midi.ui.history import Command, History
from snapmap_midi.ui.session import Session

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY_MIDI = str(FIXTURES / "tiny.mid")


def _midi(tmp_path, messages, name="song.mid") -> str:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.extend(messages)
    path = tmp_path / name
    mid.save(str(path))
    return str(path)


#: A document that touches as many levers as one song can hold at once, so the
#: byte gate below is not passing because everything in it happens to be off.
def _busy_document() -> dict:
    return settings_module.merge(
        settings_module.defaults(TINY_MIDI),
        {
            "drums": "on",
            "channels": {
                "0:0": {
                    "family": "ins_marimba",
                    "volume_db": -4,
                    "pitch_transpose": 5,
                    "polyphony": 3,
                    "voices": 2,
                    "sustain_ms": 900,
                    "release_s": 0.4,
                    "hard_stop": True,
                    "note_off": True,
                    "note_off_floor_ms": 120,
                    "attack_ms": 30,
                    "key_range": [40, 90],
                    "pitch_octave": 1,
                },
                "0:1": {
                    "sound": "play_pianoc4",
                    "pitch_follow": True,
                    "root_midi": 60,
                    "root_source": "manual",
                    "root_confidence": 1.0,
                    "fine_tune_cents": 12,
                    "glide_ms": 80,
                    "soloed": True,
                },
                "0:9": {"percussion": "kit"},
            },
            "notes": {
                "0:60:1": {"pitch_offset": 2, "volume_db": -6},
                "1:67:1": {"follow_pitch_semitones": -3},
            },
            "drum_keys": {"36": "play_drum_kick"},
            "tuning": {
                "master_volume_db": -2,
                "max_speakers": 12,
                "song_polyphony": 9,
                "release_s": 0.25,
                "hard_stop": False,
                "note_off": True,
                "note_off_floor_ms": 200,
                "max_poly": 6,
                "cap_sustain_ms": 1500,
                "bass_pitch": 70,
                "bass_cap_ms": 400,
                "decaying_families": ["ins_violin"],
                "family_caps": {"ins_flute": 350},
            },
        },
    )


# ---- importing ----


def test_a_midi_file_becomes_tracks_of_written_notes():
    song = importer.import_song(TINY_MIDI)
    # Provenance is per track: a project can hold parts from several files, so
    # no single one of them is "the" song's file.
    assert [track.source_midi for track in song.tracks] == [TINY_MIDI] * 3
    assert song.source_midis == [TINY_MIDI]
    assert song.origin == TINY_MIDI
    assert [track.key for track in song.tracks] == ["0:0", "0:1", "0:9"]
    # In the order a lane reads, not the order the file's note-offs arrived in.
    assert [(note.pitch, note.start_ms) for note in song.tracks[1].notes] == [
        (67, 500),
        (48, 1000),
    ]
    piano = song.tracks[0].notes[0]
    assert (piano.pitch, piano.velocity, piano.start_ms, piano.duration_ms) == (60, 64, 0, 500)
    assert piano.id == "0:60:1"


def test_the_song_keeps_the_clock_and_the_length_the_file_had():
    song = importer.import_song(TINY_MIDI)
    assert song.duration_ms > 0
    assert song.timing["base_bpm"] == 120.0
    assert song.tempo_map[0]["tick"] == 0
    assert song.time_signature_map[0]["numerator"] == 4


def test_the_song_gets_a_loop_spanning_its_whole_length_at_import():
    """The brace always exists, even before anyone has dragged it -- the whole
    song is the most visible, most grabbable default."""
    song = importer.import_song(TINY_MIDI)
    assert song.loop_start_ms == 0
    assert song.loop_end_ms == song.duration_ms
    assert song.loop_enabled is False


# ---- exporting the loop region ----


def test_loop_window_keeps_clips_and_rebases_notes_at_the_window_edges():
    song = importer.import_song(TINY_MIDI)
    windowed = loop_window(song, 400, 1100)

    assert windowed is not song
    # Pure: the song this session has open is never touched.
    assert song.tracks[0].notes[0].start_ms == 0
    assert song.duration_ms != 700

    assert windowed.duration_ms == 700
    assert windowed.loop_start_ms == 0
    assert windowed.loop_end_ms == windowed.duration_ms
    assert windowed.loop_enabled is False

    notes = {note.id: note for note in windowed.notes}
    # 9:36:1 (1250-1375) starts at or past the window's end and is dropped
    # entirely; the other three all overlap the window in some way.
    assert set(notes) == {"0:60:1", "1:67:1", "1:48:1"}
    # 0:60:1 (0-500) is still sounding when the window opens: clipped to the
    # start and rebased to 0.
    assert (notes["0:60:1"].start_ms, notes["0:60:1"].duration_ms) == (0, 100)
    # 1:67:1 (500-1000) sits entirely inside the window: rebased, not clipped.
    assert (notes["1:67:1"].start_ms, notes["1:67:1"].duration_ms) == (100, 500)
    # 1:48:1 (1000-1250) is still sounding when the window closes: clipped to
    # the end.
    assert (notes["1:48:1"].start_ms, notes["1:48:1"].duration_ms) == (600, 100)


def test_loop_window_carries_every_track_lever_unchanged():
    song = project.open_midi(TINY_MIDI, _busy_document())
    windowed = loop_window(song, 0, 300)
    before = {track.key: track.family for track in song.tracks}
    after = {track.key: track.family for track in windowed.tracks}
    assert after == before
    assert windowed.conversion == song.conversion


def test_the_loop_fields_round_trip_through_the_project_file(tmp_path):
    song = importer.import_song(TINY_MIDI)
    song.loop_start_ms, song.loop_end_ms, song.loop_enabled = 200, 900, True
    path = project.save(song, tmp_path / "song.smsong.json")
    reopened = project.load(path)
    assert (reopened.loop_start_ms, reopened.loop_end_ms, reopened.loop_enabled) == (
        200,
        900,
        True,
    )


def test_a_muted_or_unplayable_note_is_still_written_into_the_song(tmp_path):
    """A song holds what was WRITTEN. Mute is a fact about playback, and a
    parser that skipped muted notes would make un-muting impossible without
    going back to the file."""
    path = _midi(
        tmp_path,
        [
            mido.Message("note_on", channel=0, note=60, velocity=100, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
        ],
    )
    doc = settings_module.merge(
        settings_module.defaults(path), {"channels": {"0:0": {"muted": True}}}
    )
    song = project.open_midi(path, doc)
    assert song.tracks[0].muted is True
    assert len(song.tracks[0].notes) == 1


def test_the_program_that_was_in_force_travels_with_the_note(tmp_path):
    """A file may change program partway through a track, so the voice that
    picks an unassigned track's family cannot live on the track."""
    path = _midi(
        tmp_path,
        [
            mido.Message("program_change", channel=0, program=0, time=0),
            mido.Message("note_on", channel=0, note=60, velocity=100, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
            mido.Message("program_change", channel=0, program=40, time=0),
            mido.Message("note_on", channel=0, note=62, velocity=100, time=0),
            mido.Message("note_off", channel=0, note=62, velocity=0, time=240),
        ],
    )
    song = importer.import_song(path)
    assert [note.program for note in song.tracks[0].notes] == [0, 40]


# ---- importing more than one file ----


def test_a_second_file_joins_as_new_tracks_without_colliding(tmp_path):
    """Two files that both wrote to channel 0 are two lanes, not one. Every id
    the second import mints has to be clear of the first import's."""
    second = _midi(
        tmp_path,
        [
            mido.Message("note_on", channel=0, note=72, velocity=100, time=0),
            mido.Message("note_off", channel=0, note=72, velocity=0, time=240),
        ],
        name="second.mid",
    )
    song = importer.import_song(TINY_MIDI)
    first_ids = {track.id for track in song.tracks}
    first_notes = {note.id for note in song.notes}

    added = importer.add_midi(song, second)

    assert len(added) == 1
    assert added[0].source_midi == second
    assert song.source_midis == [TINY_MIDI, second]
    assert added[0].id not in first_ids
    assert added[0].key not in {"0:0", "0:1", "0:9"}
    assert {note.id for note in added[0].notes}.isdisjoint(first_notes)
    assert len({track.id for track in song.tracks}) == len(song.tracks)
    assert len({track.key for track in song.tracks}) == len(song.tracks)


def test_a_repeated_note_continues_its_id_series_rather_than_restarting(tmp_path):
    """The id is `channel:pitch:occurrence`, and the occurrence has to keep
    counting or the second file's first middle C would be the first file's."""
    path = _midi(
        tmp_path,
        [
            mido.Message("note_on", channel=0, note=60, velocity=100, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
        ],
        name="one-note.mid",
    )
    song = importer.import_song(path)
    assert song.tracks[0].notes[0].id == "0:60:1"

    added = importer.add_midi(song, path)
    assert added[0].notes[0].id == "0:60:2"


def test_importing_one_file_into_nothing_mints_the_ids_a_sidecar_already_names():
    """The offsets that keep a second import clear must all be zero for the
    first one, or every settings file on disk stops finding its notes."""
    song = importer.import_song(TINY_MIDI)
    assert {note.id for note in song.notes} == {"0:60:1", "1:67:1", "1:48:1", "9:36:1"}
    assert [track.part for track in song.tracks] == [(0, 0), (0, 1), (0, 9)]


# ---- migrating a settings document ----


def test_every_lever_a_document_holds_reaches_the_track_it_governs():
    doc = _busy_document()
    song = project.open_midi(TINY_MIDI, doc)
    marimba = song.track_for_part(0, 0)
    piano = song.track_for_part(0, 1)

    assert marimba.family == "ins_marimba"
    assert marimba.volume_db == -4
    assert marimba.pitch_transpose == 5
    assert (marimba.polyphony, marimba.voices) == (3, 2)
    assert marimba.sustain_ms == 900
    assert (marimba.release_s, marimba.hard_stop) == (0.4, True)
    assert (marimba.note_off, marimba.note_off_floor_ms) == (True, 120)
    assert marimba.attack_ms == 30
    assert marimba.key_range == [40, 90]
    assert marimba.pitch_octave == 1

    assert piano.sound == "play_pianoc4"
    assert piano.pitch_follow is True
    assert piano.root_midi == 60
    assert piano.fine_tune_cents == 12
    assert piano.glide_ms == 80
    assert piano.soloed is True

    assert song.track_for_part(0, 9).percussion == "kit"


def test_every_note_override_reaches_the_note_it_was_written_for():
    song = project.open_midi(TINY_MIDI, _busy_document())
    notes = {note.id: note for note in song.notes}
    assert notes["0:60:1"].pitch_offset == 2
    assert notes["0:60:1"].volume_db == -6
    assert notes["1:67:1"].follow_pitch_semitones == -3
    assert notes["1:48:1"].pitch_offset == 0.0


def test_every_conversion_lever_survives_the_migration():
    doc = _busy_document()
    song = project.open_midi(TINY_MIDI, doc)
    for lever, value in doc["tuning"].items():
        assert song.conversion[lever] == value, lever
    assert song.conversion["drums"] == "on"
    assert song.conversion["drum_keys"] == {"36": "play_drum_kick"}


def test_the_song_and_the_document_ask_the_compiler_for_the_same_thing():
    """The two lever builders sit on opposite sides of the refactor. Resolved
    per part, they have to agree on every value or the window exports something
    other than what the command line does."""
    doc = _busy_document()
    song = project.open_midi(TINY_MIDI, doc)
    from_doc = settings_module.to_compile_kwargs(doc)
    from_song = compile_levers(song)

    for keyword, expected in from_doc.items():
        if keyword == "button_name":
            # Accepted for API compatibility and then ignored: an export always
            # labels its interactive with the song's own filename.
            continue
        actual = from_song[keyword]
        if not isinstance(expected, dict) or not keyword.startswith(("part_", "channel_")):
            assert actual == expected, keyword
            continue
        for track in song.tracks:
            assert for_part(actual, *track.part) == for_part(expected, *track.part), "%s on %s" % (
                keyword,
                track.key,
            )


def test_a_wildcard_channel_setting_lands_on_every_track_it_covered(tmp_path):
    """A bare channel number means "every part on this channel". A song has no
    such key, so import resolves the value onto each track individually -- the
    same conversion, spelled out. The live LINK is what does not carry over."""
    mid = mido.MidiFile()
    for pitch in (60, 67):
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", channel=0, note=pitch, velocity=100, time=0))
        track.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=240))
        mid.tracks.append(track)
    path = str(tmp_path / "two-parts.mid")
    mid.save(path)

    doc = settings_module.merge(
        settings_module.defaults(path), {"channels": {"0": {"family": "ins_marimba"}}}
    )
    song = project.open_midi(path, doc)

    assert len(song.tracks) == 2
    assert [track.family for track in song.tracks] == ["ins_marimba", "ins_marimba"]

    song.tracks[0].family = "ins_flute"
    assert song.tracks[1].family == "ins_marimba"


def test_a_named_part_still_beats_the_wildcard_it_sits_under(tmp_path):
    mid = mido.MidiFile()
    for pitch in (60, 67):
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", channel=0, note=pitch, velocity=100, time=0))
        track.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=240))
        mid.tracks.append(track)
    path = str(tmp_path / "two-parts.mid")
    mid.save(path)

    doc = settings_module.merge(
        settings_module.defaults(path),
        {"channels": {"0": {"family": "ins_marimba"}, "1:0": {"family": "ins_flute"}}},
    )
    song = project.open_midi(path, doc)
    assert [track.family for track in song.tracks] == ["ins_marimba", "ins_flute"]


# ---- the byte gate ----


def test_an_unedited_import_compiles_to_the_bytes_the_parser_produced():
    song = importer.import_song(TINY_MIDI)
    through_song, song_stats = compile_song(song, levers=compile_levers(song))
    through_file, file_stats = compile_to_rawmap(TINY_MIDI)
    assert through_song == through_file
    assert song_stats == file_stats


def test_a_fully_tuned_song_compiles_to_the_bytes_the_document_produced():
    doc = _busy_document()
    song = project.open_midi(TINY_MIDI, doc)
    through_song, song_stats = compile_song(song, levers=compile_levers(song))
    through_file, file_stats = compile_to_rawmap(
        TINY_MIDI, **settings_module.to_compile_kwargs(doc)
    )
    assert through_song == through_file
    assert song_stats == file_stats


def test_the_session_exports_what_the_command_line_exports():
    session = Session(midi=TINY_MIDI)
    session.apply(_busy_document())
    expected, expected_stats = compile_to_rawmap(
        TINY_MIDI, **settings_module.to_compile_kwargs(session.settings())
    )
    raw, stats = session.compile()
    assert raw == expected
    assert stats == expected_stats


def test_the_command_line_entry_point_keeps_its_shape():
    """`cli.py` splats its flags into this call and is contract-tested on the
    bytes. The song model is underneath it, not in front of it."""
    raw, stats = compile_to_rawmap(TINY_MIDI, max_speakers=8, release_s=0.3, drums="off")
    assert raw.startswith(b"{")
    assert stats["max_speakers"] == 8


# ---- the project file ----


def test_a_project_round_trips_through_its_own_file(tmp_path):
    song = project.open_midi(TINY_MIDI, _busy_document())
    path = project.save(song, tmp_path / "song.smsong.json")
    reopened = project.load(path)

    assert to_dict(reopened) == to_dict(song)
    assert compile_levers(reopened) == compile_levers(song)


def test_saving_a_project_never_writes_to_the_midi(tmp_path):
    source = _midi(
        tmp_path,
        [
            mido.Message("note_on", channel=0, note=60, velocity=100, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=240),
        ],
    )
    before = Path(source).read_bytes()
    session = Session(midi=source)
    session.save_project()
    assert Path(source).read_bytes() == before
    assert project.project_path(source).is_file()


def test_a_reopened_project_compiles_to_the_same_bytes(tmp_path):
    session = Session(midi=TINY_MIDI)
    session.apply(_busy_document())
    expected, _ = session.compile()
    path = session.save_project(tmp_path / "song.smsong.json")

    reopened = Session()
    reopened.load_project(path)
    assert reopened.compile()[0] == expected


def test_a_reopened_project_describes_the_same_parts(tmp_path):
    session = Session(midi=TINY_MIDI)
    session.apply(_busy_document())
    expected = session.analysis_dict()
    path = session.save_project(tmp_path / "song.smsong.json")

    reopened = Session()
    reopened.load_project(path)
    assert reopened.analysis_dict() == expected


def test_a_project_from_a_future_build_is_refused_rather_than_half_read(tmp_path):
    payload = to_dict(importer.import_song(TINY_MIDI))
    payload["version"] = SONG_VERSION + 1
    with pytest.raises(SongError, match="version"):
        from_dict(payload)


def test_a_project_naming_a_field_this_build_does_not_know_is_refused():
    payload = to_dict(importer.import_song(TINY_MIDI))
    payload["tracks"][0]["swing"] = True
    with pytest.raises(SongError, match="swing"):
        from_dict(payload)


def test_a_missing_project_file_names_itself(tmp_path):
    with pytest.raises(SongError, match="nope.smsong.json"):
        project.load(tmp_path / "nope.smsong.json")


# ---- undo and redo ----


def test_an_empty_history_undoes_and_redoes_nothing():
    history = History()
    assert history.undo() is None
    assert history.redo() is None
    assert not history.can_undo and not history.can_redo


def test_a_pushed_command_undoes_and_redoes_in_order():
    log = []
    history = History()
    for step in ("one", "two"):
        history.push(
            Command(
                label=step,
                apply=lambda step=step: log.append("do " + step),
                revert=lambda step=step: log.append("undo " + step),
            )
        )

    assert history.undo_label == "two"
    assert history.undo() == "two"
    assert history.undo() == "one"
    assert history.undo() is None
    assert history.redo() == "one"
    assert history.redo() == "two"
    assert log == ["undo two", "undo one", "do one", "do two"]


def test_a_new_command_discards_whatever_was_waiting_to_be_redone():
    history = History()
    history.push(Command("first", lambda: None, lambda: None))
    history.undo()
    history.push(Command("second", lambda: None, lambda: None))
    assert not history.can_redo
    assert history.undo_label == "second"


def test_the_history_forgets_the_oldest_step_past_its_limit():
    history = History(limit=2)
    for step in ("a", "b", "c"):
        history.push(Command(step, lambda: None, lambda: None))
    assert history.undo() == "c"
    assert history.undo() == "b"
    assert history.undo() is None


def test_a_session_undoes_a_change_pushed_onto_it():
    session = Session(midi=TINY_MIDI)
    song = session.song()
    note = song.tracks[0].notes[0]
    original = note.start_ms
    note.start_ms = original + 250
    session.push_command(
        "Move note",
        revert=lambda: setattr(note, "start_ms", original),
        apply=lambda: setattr(note, "start_ms", original + 250),
    )

    assert session.undo()["undone"] == "Move note"
    assert note.start_ms == original
    assert session.redo()["redone"] == "Move note"
    assert note.start_ms == original + 250


def test_the_bridge_answers_rather_than_raising_for_every_project_call(tmp_path):
    """The window has nowhere to put an exception: it arrives in Javascript as
    an opaque Error and the button simply goes dead."""
    from snapmap_midi.ui.api import Bridge

    empty = Bridge()
    assert empty.save_project()["ok"] is False
    assert "error" in empty.save_project()
    assert empty.load_project(tmp_path / "absent.smsong.json")["ok"] is False
    assert empty.undo()["ok"] is True
    assert empty.redo()["ok"] is True


def test_the_bridge_saves_and_reopens_a_project(tmp_path):
    from snapmap_midi.ui.api import Bridge

    # A copy, not the fixture: `apply_settings` autosaves a sidecar beside the
    # song, and `tests/fixtures/` is input.
    song_path = tmp_path / "tiny.mid"
    song_path.write_bytes(Path(TINY_MIDI).read_bytes())

    bridge = Bridge(midi=str(song_path))
    bridge.apply_settings({"channels": {"0:0": {"family": "ins_marimba"}}})
    saved = bridge.save_project(tmp_path / "song.smsong.json")
    assert saved["ok"] is True

    reopened = Bridge()
    payload = reopened.load_project(saved["project"])
    assert payload["ok"] is True
    assert payload["settings"]["channels"]["0:0"]["family"] == "ins_marimba"
    assert payload["analysis"]["channels"][0]["key"] == "0:0"


def test_opening_a_song_starts_a_clean_history():
    session = Session(midi=TINY_MIDI)
    session.push_command("Move note", revert=lambda: None)
    assert session.history_state()["can_undo"] is True
    session.load(TINY_MIDI)
    assert session.history_state()["can_undo"] is False
