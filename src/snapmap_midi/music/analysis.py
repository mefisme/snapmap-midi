"""What a MIDI file contains, named in the user's terms rather than the compiler's.

`parse_notes` cannot answer this. By the time it returns, a program number has
become a family and the channel's own identity -- which instrument the composer
asked for -- is gone. Choosing an instrument per channel needs the question
asked before that collapse, so this reads the file separately.

It keeps the per-note histogram rather than only the extremes. The extremes
alone cannot answer the question the window exists to answer -- how many notes a
chosen family cannot reach -- and they draw the same bar for two notes as for
two thousand, so one stray low note makes a piano part look like it spans the
keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from snapmap_midi.music.gm import (
    drum_table,
    gm_drum_kit_name,
    gm_drum_name,
    gm_program_name,
    gm_to_family,
)
from snapmap_midi.music.midi import (
    channel_is_percussion,
    is_percussion_part,
    messages_with_tracks,
)


@dataclass
class ChannelInfo:
    """One channel, described the way the window's row for it has to describe it.

    `pitches` is the whole histogram rather than a summary because it is what
    every later answer is computed from: the out-of-range count, the density
    strip, and the drum-key list. Revision 1 built exactly this table inside
    `analyze` and kept only `lowest` and `highest`, which left the design's
    headline number uncomputable and drew a part's two hundred middle-C notes
    and its one stray bottom A as the same undifferentiated bar.

    `drum_keys` is empty unless `is_drums`, and a `None` value means the key is
    one `DRUM_MAP` has no sound for -- the case a channel assignment or an
    advanced per-key sidecar override can fill.
    """

    channel: int
    program: int
    program_name: str
    notes: int
    lowest: Optional[int]
    highest: Optional[int]
    is_drums: bool
    auto_family: Optional[str]
    pitches: dict
    drum_keys: dict
    track: int = 0
    track_name: str = ""
    #: The `Track.id` this channel was read from, or "" when there is none to
    #: read (the file-based `analyze` has no `Song` and therefore no track
    #: identity). `from_song` always sets this; it is what lets the window
    #: draw a note onto a track that has no notes yet at all -- a hand-drawn
    #: note has no existing preview event to read a `track_id` off, so the
    #: channel/part payload is where it has to come from instead.
    track_id: str = ""
    #: The file a track was imported from, or None for one drawn from
    #: nothing. Carried here so the window can show or hide a per-track
    #: "Reopen from source" action without a second round trip.
    source_midi: Optional[str] = None

    @property
    def key(self) -> str:
        """This part's identity: the track that wrote it and the channel it uses.

        A bare channel number is not an identity. Two tracks sharing channel 0 --
        a lead and a pad, which is ordinary in a type 1 file -- are two parts a
        composer chose separately, and keying anything by channel alone merges
        them into one row that can hold only one instrument.

        The track comes first so parts sort in the order the file lists them,
        which is the order the composer sees in their editor.
        """
        return "%d:%d" % (self.track, self.channel)


@dataclass
class MidiAnalysis:
    """A whole file, plus the one judgement that had to be made while reading it.

    `drums_detected` is recorded rather than recomputed on demand because the
    heuristic behind it walks the entire file, and because the window shows it
    as the position of a switch. A switch whose position is derived twice can
    disagree with itself between the row that draws it and the compile that
    obeys it.
    """

    path: str
    duration_s: float
    drums_detected: bool
    channels: list


def analyze(mid_path, drums="auto", part_percussion=None) -> MidiAnalysis:
    """Read a file's channels without collapsing them into families.

    Takes the drums mode because the window's drums switch decides whether
    channel 9 is a kit. An analysis cached from `"auto"` would keep offering a
    family dropdown for a channel the compiler had since started routing
    through `DRUM_MAP`, so the row would describe an instrument nothing plays.

    The mode is resolved to match `parse_notes`, and that sameness is the
    point: if these two ever disagree, the window describes one arrangement
    while the compiler writes another.

    The settings document spells the mode `"auto"`, `"on"` or `"off"`, while
    `parse_notes` takes `"auto"` or a bool. Both spellings are accepted here
    because a bare `bool("off")` is True -- the string that means "no drums"
    would have forced them on, and the row for a silenced kit would have gone on
    listing the keys it was playing.
    """
    import mido

    part_percussion = part_percussion or {}
    mid = mido.MidiFile(str(mid_path), clip=True)
    if drums == "auto":
        drums_on = channel_is_percussion(mid)
    elif isinstance(drums, str):
        drums_on = drums == "on"
    else:
        drums_on = bool(drums)

    # SMF FF 03: "If in a format 0 track, or the first track in a format 1
    # file, the name of the sequence. Otherwise, the name of the track." So the
    # one name a format 0 file carries is the SONG's title, and using it as a
    # part label would print the song's name over every row. Format 1's first
    # track is the conductor and holds the sequence name for the same reason;
    # it has no notes, so it never becomes a part, but the rule is spelled out
    # here rather than left to that coincidence.
    names = {}
    for index, track in enumerate(mid.tracks):
        if mid.type == 0 or index == 0:
            names[index] = ""
            continue
        named = next((m.name for m in track if m.type == "track_name"), "")
        names[index] = named.strip()

    # Program changes are channel messages, so a later one anywhere in the file
    # is the channel's current voice. But a type 1 file normally has each track
    # announce its own instrument, and reading a neighbouring track's program
    # would rename a part the composer never touched. Prefer what the part's own
    # track said; fall back to the channel only when it said nothing.
    by_part, by_channel, seen, elapsed = {}, {}, {}, 0.0
    for track_index, msg, elapsed in messages_with_tracks(mid):
        if msg.type == "program_change":
            by_channel[msg.channel] = msg.program
            by_part[(track_index, msg.channel)] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            part = (track_index, msg.channel)
            entry = seen.setdefault(
                part,
                {
                    "program": by_part.get(part, by_channel.get(msg.channel, 0)),
                    "pitches": {},
                },
            )
            entry["pitches"][msg.note] = entry["pitches"].get(msg.note, 0) + 1

    # Read once. `drum_table` opens the user's file, and a song with twenty
    # percussion parts would otherwise open it twenty times.
    table = drum_table()
    channels = []
    for track_index, channel in sorted(seen):
        entry = seen[(track_index, channel)]
        pitches = entry["pitches"]
        is_drums = is_percussion_part(part_percussion, track_index, channel, drums_on)
        channels.append(
            ChannelInfo(
                channel=channel,
                program=entry["program"],
                # On the percussion channel the program selects a kit, not an
                # instrument, so the melodic name is simply the wrong table.
                program_name=(
                    gm_drum_kit_name(entry["program"])
                    if is_drums
                    else gm_program_name(entry["program"])
                ),
                notes=sum(pitches.values()),
                lowest=min(pitches),
                highest=max(pitches),
                is_drums=is_drums,
                auto_family=None if is_drums else gm_to_family(entry["program"]),
                pitches=pitches,
                drum_keys={k: table.get(k) for k in sorted(pitches)} if is_drums else {},
                track=track_index,
                track_name=names.get(track_index, ""),
            )
        )
    return MidiAnalysis(str(mid_path), round(elapsed, 2), drums_on, channels)


def from_song(song) -> MidiAnalysis:
    """Describe an OPEN SONG the same way `analyze` describes a file.

    Same answer, asked of the thing the window is actually editing. Once a song
    is imported the file has told us everything it is going to: re-reading it to
    redraw a row would describe the file rather than the song, and after a note
    is moved or a track is added those are two different arrangements. It also
    means a saved project opens with no `.mid` present at all.

    The drums switch and each track's percussion mode both live on the song, so
    they are read from it rather than passed in -- the row that draws a kit and
    the compile that routes it through `DRUM_MAP` cannot disagree if there is
    only one place either can look.
    """
    from snapmap_midi.music.midi import written_is_percussion

    declared = song.conversion.get("drums", "auto")
    if declared == "auto":
        # Asked of the song's own written notes rather than of a file, so a
        # project with no `.mid` beside it still answers.
        drums_on = written_is_percussion(
            (track.channel, note.pitch) for track in song.tracks for note in track.notes
        )
    else:
        drums_on = declared == "on"

    # Read once. `drum_table` opens the user's file, and a song with twenty
    # percussion parts would otherwise open it twenty times.
    table = drum_table()
    channels = []
    for track in sorted(song.tracks, key=lambda t: t.part):
        # A track with no notes yet -- freshly drawn, or every note deleted --
        # still gets a row. `analysis.channels` is the ONLY thing the whole
        # frontend track column iterates (`buildTracks`/`buildLanesView`), so
        # skipping it here would make the track unselectable, unable to take
        # a sound, and unable to ever receive a drawn note: exactly the
        # compose-from-nothing case this phase exists to support.
        pitches: dict = {}
        for note in track.notes:
            pitches[note.pitch] = pitches.get(note.pitch, 0) + 1
        # An empty track has no note to read a program off; 0 (Acoustic Grand
        # Piano) is the same silent default a channel nobody has touched gets
        # anywhere else in this module.
        program = track.notes[0].program if track.notes else 0
        is_drums = is_percussion_part(
            {track.part: track.percussion}, track.source_track, track.channel, drums_on
        )
        channels.append(
            ChannelInfo(
                channel=track.channel,
                program=program,
                # On the percussion channel the program selects a kit, not an
                # instrument, so the melodic name is simply the wrong table.
                program_name=(gm_drum_kit_name(program) if is_drums else gm_program_name(program)),
                notes=sum(pitches.values()),
                # None rather than a number for an empty track: there is no
                # lowest or highest note, and `min`/`max` on an empty
                # histogram would raise rather than answer that. Every
                # consumer of these two fields already treats `None` as a
                # real, expected answer -- `ChannelInfo`'s own typing has
                # always said `Optional[int]`, and `Session.channel_info`
                # already raises on exactly this pair for a different reason
                # (nothing to pitch-anchor a sound choice against).
                lowest=min(pitches) if pitches else None,
                highest=max(pitches) if pitches else None,
                is_drums=is_drums,
                auto_family=None if is_drums else gm_to_family(program),
                pitches=pitches,
                drum_keys={k: table.get(k) for k in sorted(pitches)} if is_drums else {},
                track=track.source_track,
                track_name=track.name,
                track_id=track.id,
                source_midi=track.source_midi,
            )
        )
    return MidiAnalysis(
        str(song.origin or ""), round(song.duration_ms / 1000.0, 2), drums_on, channels
    )


def as_dict(analysis: MidiAnalysis) -> dict:
    """The analysis as JSON, for the one consumer that cannot take it any other way.

    `pitches` and `drum_keys` come back with string keys because JSON has no
    integer ones. Converted here rather than left to whatever unpacks the
    payload, because the failure is silent in both directions: a reader that
    assumes integers looks up note 60, finds nothing, and draws an empty ruler
    without ever raising.
    """
    return {
        "path": analysis.path,
        "duration_s": analysis.duration_s,
        "drums_detected": analysis.drums_detected,
        "channels": [
            {
                "key": c.key,
                "track": c.track,
                "track_name": c.track_name,
                "track_id": c.track_id,
                "source_midi": c.source_midi,
                "channel": c.channel,
                "program": c.program,
                "program_name": c.program_name,
                "notes": c.notes,
                "lowest": c.lowest,
                "highest": c.highest,
                "is_drums": c.is_drums,
                "auto_family": c.auto_family,
                "pitches": {str(note): count for note, count in c.pitches.items()},
                "drum_keys": {str(key): shader for key, shader in c.drum_keys.items()},
                # Carried beside the keys rather than looked up from the
                # catalog, because the catalog is built when a file opens and
                # this list changes whenever a part is declared a kit. A name
                # fetched from the stale one reads every key as unnamed.
                "drum_names": {str(key): gm_drum_name(key) for key in c.drum_keys},
            }
            for c in analysis.channels
        ],
    }


def notes_outside(channel: ChannelInfo, span) -> int:
    """How many of a channel's notes a family cannot reach.

    `decl_for` never fails outside the range -- it prefers the same pitch class
    an octave away, and falls back to the nearest available pitch when the class
    is absent entirely. So this is not a count of dropped notes; it is a count
    of notes that will not be the note that was written. That distinction is
    why the warning says "move to another octave" rather than "are lost".
    """
    if span is None:
        return 0
    low, high = span
    return sum(n for note, n in channel.pitches.items() if note < low or note > high)


def ruler_segments(channel: ChannelInfo, span, axis) -> Optional[dict]:
    """Where to draw a channel's notes and its instrument's reach, in percent.

    In Python rather than Javascript because every failure this geometry can
    have -- an axis that clips the family it was built to showcase, a disjoint
    range drawn as a matter of degree, a drum channel plotted on a pitch axis --
    is a failure a test can catch here and nothing can catch there.

    Returns None for the percussion channel: its lowest and highest are key
    numbers, and drawing them on a pitch axis asserts something false.
    """
    if channel.is_drums or not channel.pitches:
        return None
    floor, ceiling = axis
    reach = float(ceiling - floor)
    heaviest = max(channel.pitches.values())

    def place(note):
        return max(0.0, min(100.0, (note - floor) / reach * 100.0))

    cells = [
        {"note": note, "left": place(note), "weight": count / heaviest}
        for note, count in sorted(channel.pitches.items())
    ]
    instrument = None
    disjoint = False
    if span is not None:
        low, high = span
        left = place(low)
        instrument = {"left": left, "width": max(0.0, place(high) - left)}
        disjoint = channel.highest < low or channel.lowest > high
    return {
        "cells": cells,
        "instrument": instrument,
        "disjoint": disjoint,
        "outside": notes_outside(channel, span),
        "cell_width": 100.0 / reach,
    }
