"""Reading a `.mid` into a song, once.

Import happens at the moment a file is opened and never again. That is the
whole change: every compile and every preview used to re-read the file, so the
notes had nowhere to live and nothing about them could be edited. Here the file
is turned into tracks and notes, and from then on the song answers for itself.

Importing is written as "read this file into a list of tracks that can join
anything", not as "build me a song". A second `.mid` dropped onto an open
project has to land as more tracks beside the ones already there, so nothing in
here may assume it is the only content: track ids, part identities and note ids
are all minted clear of whatever the caller already holds.

Nothing here knows what a settings document is. A song arrives with no levers
set, and `snapmap_midi.project` -- which sits above both this and the settings
document -- is what puts them on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from snapmap_midi.music import timing as timing_module
from snapmap_midi.music.midi import pair_notes
from snapmap_midi.music.song import Note, Song, Track

#: An imported note id: the channel, the written pitch, and which time that
#: pairing has been seen. Matched rather than assumed so a song that also holds
#: notes drawn from nothing -- which carry a different id shape entirely -- can
#: still be counted for the next import's numbering.
_IMPORTED_NOTE_ID = re.compile(r"^(\d+):(\d+):(\d+)$")


@dataclass
class Import:
    """One file read into a form that can join a song, or start one."""

    tracks: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    duration_ms: int = 0


def _track_names(mid) -> dict:
    """Each SMF track's own name, or empty where the name is the song's.

    SMF FF 03: "If in a format 0 track, or the first track in a format 1 file,
    the name of the sequence. Otherwise, the name of the track." So the one name
    a format 0 file carries is the SONG's title, and using it as a track label
    would print the song's name over every lane.
    """
    names = {}
    for index, track in enumerate(mid.tracks):
        if mid.type == 0 or index == 0:
            names[index] = ""
            continue
        named = next((m.name for m in track if m.type == "track_name"), "")
        names[index] = named.strip()
    return names


def _occurrences_in_use(existing_tracks) -> dict:
    """How far each `(channel, pitch)` id series has already been counted."""
    seen: dict = {}
    for track in existing_tracks:
        for note in track.notes:
            match = _IMPORTED_NOTE_ID.match(note.id)
            if match is None:
                continue
            channel, pitch, occurrence = (int(part) for part in match.groups())
            key = (channel, pitch)
            seen[key] = max(seen.get(key, 0), occurrence)
    return seen


def import_tracks(mid_path, *, mid=None, existing_tracks=()) -> Import:
    """Read one `.mid` into tracks that can be appended to `existing_tracks`.

    Pure with respect to where the result is going: it takes what is already
    held rather than a song, so seeding a new project and adding a file to an
    open one are the same call with a different argument.

    Nothing minted here can collide with what was already there. Track ids are
    taken clear of the ids in use; each imported part's `(source_track,
    channel)` identity is offset past every part already present, so two files
    that both wrote to channel 0 stay two separate lanes; and note ids continue
    each `(channel, pitch)` series rather than restarting it. For the ordinary
    case -- the first file into an empty project -- every one of those offsets
    is zero, so the ids are exactly the ones a settings sidecar written by an
    earlier build already names.
    """
    import mido

    if mid is None:
        # clip=True clamps out-of-range data bytes rather than refusing the file.
        mid = mido.MidiFile(str(mid_path), clip=True)

    existing_tracks = list(existing_tracks)
    track_offset = (
        max(track.source_track for track in existing_tracks) + 1 if existing_tracks else 0
    )
    taken_ids = {track.id for track in existing_tracks}
    occurrences = _occurrences_in_use(existing_tracks)

    source_notes, end_ms, _elapsed = pair_notes(mid)
    names = _track_names(mid)

    by_part: dict = {}
    for source in source_notes:
        part = (source.track + track_offset, source.channel)
        match = _IMPORTED_NOTE_ID.match(source.id)
        occurrence = int(match.group(3)) + occurrences.get((source.channel, source.pitch), 0)
        note = _note_from_source(source, "%d:%d:%d" % (source.channel, source.pitch, occurrence))
        by_part.setdefault(part, []).append(note)

    tracks = []
    serial = len(existing_tracks)
    for source_track, channel in sorted(by_part):
        serial += 1
        while ("t:%d" % serial) in taken_ids:
            serial += 1
        track_id = "t:%d" % serial
        taken_ids.add(track_id)
        tracks.append(
            Track(
                id=track_id,
                name=names.get(source_track - track_offset, ""),
                channel=channel,
                source_track=source_track,
                source_midi=str(mid_path),
                # Pairing hands these back in note-OFF order, which is the order
                # they finish in. A lane reads left to right, so they are stored
                # the way a roll draws them. Nothing downstream depends on
                # either: the compiler and the preview both re-sort before a
                # byte is written or a note is scheduled.
                notes=sorted(
                    by_part[(source_track, channel)],
                    key=lambda note: (note.start_ms, note.pitch, note.id),
                ),
            )
        )
    return Import(tracks=tracks, timing=timing_module.manifest(mid), duration_ms=end_ms)


def _note_from_source(source, note_id: str) -> Note:
    return Note(
        id=note_id,
        pitch=source.pitch,
        velocity=source.velocity,
        start_ms=source.start,
        duration_ms=source.end - source.start,
        program=source.program,
    )


def import_song(mid_path, *, mid=None) -> Song:
    """Start a project from one `.mid`, with nothing chosen about it yet.

    Levers arrive separately, through `snapmap_midi.project`, because they come
    from a settings document and this layer must stay usable without one.
    """
    result = import_tracks(mid_path, mid=mid)
    return Song(
        duration_ms=result.duration_ms,
        timing=result.timing,
        tracks=result.tracks,
    )


def add_midi(song: Song, mid_path, *, mid=None) -> list:
    """Add another `.mid` to an open project as further tracks. Returns them.

    The song's own clock is the first file's: a project has one ruler, and a
    later file's tempo map cannot retroactively move the bar lines under music
    already written. Its length grows to hold whatever arrives.
    """
    result = import_tracks(mid_path, mid=mid, existing_tracks=song.tracks)
    song.tracks.extend(result.tracks)
    song.duration_ms = max(song.duration_ms, result.duration_ms)
    if not song.timing:
        song.timing = result.timing
    return result.tracks
