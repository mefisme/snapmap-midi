"""Writing a `Song` back out as a standard `.mid` file.

The inverse of `music/importer.py`, and deliberately narrower than it: import
turns a file into everything a song can hold, and this turns a song back into
everything a `.mid` file CAN hold -- notes, track structure, and the
tempo/time-signature map. A DOOM sound assignment or a SnapMap pitch/volume
expression has no field in a Standard MIDI File to carry it in, by design (see
the plan's own reasoning: this is the "share with anyone, any DAW" path, not a
second project format). The project file (`snapmap_midi.project.save`) is what
keeps everything; this keeps what MIDI itself can mean.

Notes are stored on `Song` in milliseconds at speed 1.0; a `.mid` file is
written in ticks. `music/timing.py::tick_at_time` is the same conversion the
piano roll's own ruler uses, so an exported note lands on the same tick the
roll would have drawn a bar line at, at that note's own start.
"""

from __future__ import annotations

from pathlib import Path

from snapmap_midi.music import timing as timing_module
from snapmap_midi.music.song import Song

#: MIDI's own default tempo and time signature, used when a song's timing has
#: recorded neither -- the same fallback `music/timing.py::manifest` reads
#: from a file that never sets either.
_DEFAULT_TEMPO_US = 500_000
_DEFAULT_SIGNATURE = {"tick": 0, "numerator": 4, "denominator": 4}


def build_midi_file(song: Song):
    """The song as an in-memory `mido.MidiFile`, format 1.

    Track 0 is the conductor track (tempo and time-signature meta events
    only, no notes) -- the SMF convention every DAW and `music/midi.py`'s own
    importer already expects. One track per `Song.track` follows, in the
    song's own order, so re-importing the result lines tracks up the same way
    the original file would have.
    """
    import mido

    timing = song.timing or {}
    ticks_per_beat = int(timing.get("ticks_per_beat") or 480)
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    conductor = mido.MidiTrack()
    mid.tracks.append(conductor)
    tempo_changes = sorted(
        song.tempo_map or [{"tick": 0, "tempo": _DEFAULT_TEMPO_US}],
        key=lambda marker: int(marker["tick"]),
    )
    signatures = sorted(
        song.time_signature_map or [_DEFAULT_SIGNATURE],
        key=lambda marker: int(marker["tick"]),
    )
    meta_events = [
        (int(marker["tick"]), 0, mido.MetaMessage("set_tempo", tempo=int(marker["tempo"]), time=0))
        for marker in tempo_changes
    ] + [
        (
            int(marker["tick"]),
            1,
            mido.MetaMessage(
                "time_signature",
                numerator=int(marker["numerator"]),
                denominator=int(marker["denominator"]),
                time=0,
            ),
        )
        for marker in signatures
    ]
    meta_events.sort(key=lambda item: (item[0], item[1]))
    previous_tick = 0
    for tick, _kind, message in meta_events:
        message.time = max(0, tick - previous_tick)
        conductor.append(message)
        previous_tick = tick
    conductor.append(mido.MetaMessage("end_of_track", time=0))

    for index, track in enumerate(song.tracks):
        midi_track = mido.MidiTrack()
        mid.tracks.append(midi_track)
        if track.name:
            midi_track.append(mido.MetaMessage("track_name", name=track.name, time=0))
        # Standard MIDI has sixteen channels; a project can hold more tracks
        # than that (every hand-drawn one mints its own `channel` counter --
        # see `Song.drawn_channel_serial` -- with no ceiling). Wrapping is an
        # honest lossy fallback rather than a refusal: two tracks sharing a
        # channel in the exported file play as two overlapping parts on one
        # channel, which is still music, just not separable by channel alone
        # the way the original project was by `Track.id`.
        channel = int(track.channel or 0) % 16

        raw_events = []
        for note in track.notes:
            pitch = max(0, min(127, int(round(note.pitch))))
            velocity = max(1, min(127, int(round(note.velocity)) or 1))
            start_tick = max(0, tick_at_time_or_default(timing, note.start_ms))
            end_tick = max(
                start_tick + 1, tick_at_time_or_default(timing, note.start_ms + note.duration_ms)
            )
            note_on = mido.Message("note_on", note=pitch, velocity=velocity, channel=channel)
            note_off = mido.Message("note_off", note=pitch, velocity=0, channel=channel)
            raw_events.append((start_tick, 1, note_on))
            raw_events.append((end_tick, 0, note_off))
        # Note-offs sort before note-ons at the same tick, so a note that
        # ends exactly when the next one starts never sounds stuck retriggered.
        raw_events.sort(key=lambda item: (item[0], item[1]))
        previous_tick = 0
        for tick, _kind, message in raw_events:
            message.time = max(0, tick - previous_tick)
            midi_track.append(message)
            previous_tick = tick
        midi_track.append(mido.MetaMessage("end_of_track", time=0))

    return mid


def tick_at_time_or_default(timing: dict, time_ms) -> int:
    """`timing_module.tick_at_time`, safe for a song whose timing is empty.

    A song opened as a brand-new project always seeds a real timing manifest
    (`Session._blank_timing`), but nothing downstream of `Song` enforces
    that, so this is the same defensive fallback `build_midi_file` would
    otherwise have to repeat at every call site.
    """
    if not timing:
        return int(round(int(time_ms) * 1000 * 480 / _DEFAULT_TEMPO_US))
    return timing_module.tick_at_time(timing, time_ms)


def write_midi(song: Song, path) -> Path:
    """Write `song` to `path` as a standard `.mid` file. Returns the path."""
    destination = Path(path)
    mid = build_midi_file(song)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(destination))
    return destination
