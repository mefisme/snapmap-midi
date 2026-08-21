"""The editable document: one song, its tracks, and the notes on them.

This is the thing the workstation edits. Before it existed there was nothing to
edit: every compile and every preview re-read the `.mid` from disk and threw the
parsed notes away, so the only state that survived a click was a sidecar of
*overrides* keyed by a derived id -- `channel:pitch:occurrence`, the Nth time
that pitch appeared on that channel. That key is a position, and a position
moves the instant a note is inserted or deleted, so every later note's settings
would follow the wrong note.

A `Song` holds the notes themselves. Ids are assigned once, at the moment a note
comes into existence, and are never recomputed from where it sits -- which is
what makes moving, resizing and deleting expressible at all.

Nothing in here reads a MIDI file. Import lives in `importer.py`, the lever
projection in `levers.py`, and the conversion itself downstream of both. This
module is the record and its JSON round-trip, so a saved project can be read
back by a build that has changed everything else.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import Optional

#: Bumped when a project file written by this build can no longer be read by
#: the previous one. `from_dict` refuses a version it does not know rather than
#: reading the half it understands, for the same reason `settings.validate`
#: does: a partial read loses exactly the fields the newer build added.
SONG_VERSION = 1

#: The conversion levers that belong to the whole song, with the values
#: `compile_to_rawmap` uses when nobody sets them. Mirrors `settings.py`'s
#: `_TUNING_DEFAULTS` plus the two song-wide percussion choices, which are
#: about this arrangement rather than about the user's setup and therefore
#: travel with the song rather than with the workstation.
DEFAULT_CONVERSION = {
    "master_volume_db": 0,
    "max_speakers": 32,
    "song_polyphony": 32,
    "release_s": 0.1,
    "hard_stop": False,
    "note_off": False,
    "note_off_floor_ms": None,
    "max_poly": None,
    "cap_sustain_ms": None,
    "bass_pitch": 78,
    "bass_cap_ms": None,
    "decaying_families": [],
    "family_caps": {},
    "playback_speed": 1.0,
    "drums": "auto",
    "drum_keys": {},
}


@dataclass
class Note:
    """One written note, plus the playback expression the user chose for it.

    `start_ms` and `duration_ms` are the song's own clock at `playback_speed`
    1.0. Speed is a conversion lever layered on top rather than a rewrite of
    what was written, so a tempo audition never has to touch note data.

    `program` is the General MIDI voice in force when this note was written. It
    is provenance rather than a lever: it is what picks a family for a track
    nobody has assigned an instrument to, and a file may change program partway
    through a track, so it cannot live on the track.
    """

    id: str
    pitch: int
    velocity: int
    start_ms: int
    duration_ms: int
    program: int = 0
    # Every per-note override the sidecar carried, none dropped. `pitch_offset`
    # stays a relative fallback for both pitch modes; `pitch_semitones` is the
    # absolute manual value and `follow_pitch_semitones` overrides the
    # automatic one only while Follow MIDI note is on. `volume_db` is absolute
    # and preserves an explicit zero, which is why it is None rather than 0
    # when unset; `volume_trim_db` is the legacy relative trim.
    pitch_offset: float = 0.0
    pitch_semitones: Optional[float] = None
    follow_pitch_semitones: Optional[float] = None
    volume_db: Optional[int] = None
    volume_trim_db: int = 0

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


#: Per-track levers, and the value each takes when the track says nothing. A
#: `None` here is not "off" -- it is "this track defers to the song", which is
#: a different answer from any number the track could hold.
_TRACK_LEVERS = {
    "family": None,
    "sound": None,
    "percussion": "auto",
    "muted": False,
    "soloed": False,
    "pitch_follow": False,
    "pitch_follow_preference": None,
    "root_midi": None,
    "detected_root_midi": None,
    "root_confidence": None,
    "root_source": None,
    "pitch_transpose": 0.0,
    "pitch_octave": None,
    "fine_tune_cents": 0.0,
    "glide_ms": None,
    "attack_ms": None,
    "release_s": None,
    "hard_stop": None,
    "note_off": None,
    "note_off_floor_ms": None,
    "sustain_ms": None,
    "volume_db": 0,
    "voices": None,
    "polyphony": None,
    "key_range": None,
}


@dataclass
class Track:
    """One lane of the arrangement, and every choice that governs how it plays.

    `source_midi`, `source_track` and `channel` are where an imported part came
    from. Provenance lives here rather than on the song because a project may
    hold tracks imported from several different files, so "which file is this
    from" is a fact about one lane and not about the arrangement. A track drawn
    from nothing carries None, and -1 for the source track, which no MIDI file
    can produce.

    `source_track` and `channel` keep their old meaning as an identity: two MIDI
    tracks writing to channel 0 are two parts a composer chose separately, so a
    channel number alone was never an identity.
    """

    id: str
    name: str = ""
    channel: int = 0
    source_track: int = -1
    source_midi: Optional[str] = None
    notes: list = field(default_factory=list)
    family: Optional[str] = None
    sound: Optional[str] = None
    percussion: str = "auto"
    muted: bool = False
    soloed: bool = False
    pitch_follow: bool = False
    pitch_follow_preference: Optional[bool] = None
    root_midi: Optional[float] = None
    detected_root_midi: Optional[float] = None
    root_confidence: Optional[float] = None
    root_source: Optional[str] = None
    pitch_transpose: float = 0.0
    pitch_octave: Optional[int] = None
    fine_tune_cents: float = 0.0
    glide_ms: Optional[int] = None
    attack_ms: Optional[int] = None
    release_s: Optional[float] = None
    hard_stop: Optional[bool] = None
    note_off: Optional[bool] = None
    note_off_floor_ms: Optional[int] = None
    sustain_ms: Optional[int] = None
    volume_db: int = 0
    voices: Optional[int] = None
    polyphony: Optional[int] = None
    key_range: Optional[list] = None

    @property
    def part(self) -> tuple:
        """The `(track, channel)` pair every per-part lever is keyed by."""
        return (self.source_track, self.channel)

    @property
    def key(self) -> str:
        """The same identity as text, which is how a settings document and the
        window's own analysis both spell a part."""
        return "%d:%d" % (self.source_track, self.channel)

    def clear_levers(self) -> None:
        """Return every lever to its unset value, keeping the notes.

        Used when the settings document is re-projected onto the song: a lever
        the user has just cleared has no entry in the patch at all, so anything
        left standing from the previous projection would survive a removal.
        """
        for name, value in _TRACK_LEVERS.items():
            setattr(self, name, copy.deepcopy(value))


@dataclass
class Song:
    """One arrangement: what it is made of, and what to convert it with.

    Provenance is per track, not here: a project may hold tracks imported from
    several files, and no single one of them is "the" song's file. What a file
    contributed is never read again for notes either way -- that is the whole
    point of holding them here.

    `timing` is the source clock as absolute tick/time change points, at speed
    1.0. The piano roll's ruler is drawn from it, and it is stored rather than
    recomputed because the file it came from may be gone by the time a project
    is reopened.
    """

    version: int = SONG_VERSION
    #: Where the song ends, in milliseconds at speed 1.0. Seeded at import from
    #: the file's own grid-completed length (see `music/importer.py`), and from
    #: then on this is the AUTHORITATIVE length -- `set_song_length` is the only
    #: thing that changes it. Nothing recomputes it from note content on its
    #: own: shrinking it past the last note's end does not delete or clip that
    #: note, it only stops the note from playing (see `beyond_length` in
    #: `music/midi.py::resolve_notes`).
    duration_ms: int = 0
    #: The loop region the ruler's brace always shows, whether or not it is
    #: doing anything. Seeded at import to span the whole song (see
    #: `music/importer.py`), same as `duration_ms` -- there is no "unset"
    #: state to fall back to, only a region that has not been dragged yet.
    #: `Session.set_loop` is what moves either bound.
    loop_start_ms: int = 0
    loop_end_ms: int = 0
    #: Whether the loop actually does anything: wraps playback at `loop_end_ms`
    #: back to `loop_start_ms`, and gates nothing else -- `export_loop` reads
    #: the bounds above regardless of this flag, since exporting a region is a
    #: different question from whether the transport is currently looping it.
    #: Off by default. `Session.set_loop_enabled` is the only thing that
    #: changes it, and it is not undo-tracked, matching mute/solo.
    loop_enabled: bool = False
    timing: dict = field(default_factory=dict)
    tracks: list = field(default_factory=list)
    conversion: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONVERSION))
    #: Hands out ids for notes that no MIDI file named. Serialized so a project
    #: reopened twice cannot mint an id it already used.
    note_serial: int = 0
    #: The highest channel number ever handed to a hand-drawn track (one with
    #: `source_track == -1`), or -1 before any exist. A hand-drawn track has no
    #: MIDI provenance, so `channel` is the ONLY thing that varies in its
    #: `key` ("-1:channel") -- and that key is what the settings document uses
    #: to keep two tracks' levers apart. Minting the next channel from the
    #: tracks CURRENTLY in the song (`max(...) + 1`) looks right until one is
    #: deleted: the max drops, and the next track created reuses the deleted
    #: one's channel number and therefore its stale, still-present settings
    #: entry. This counter never drops, so a channel is never handed out
    #: twice regardless of what has been deleted in between. Serialized for
    #: the same reason `note_serial` is.
    drawn_channel_serial: int = -1

    @property
    def tempo_map(self) -> list:
        return list(self.timing.get("tempo_changes", ()))

    @property
    def time_signature_map(self) -> list:
        return list(self.timing.get("time_signatures", ()))

    @property
    def source_midis(self) -> list:
        """Every file this song was imported from, in the order tracks joined.

        More than one is ordinary: importing a second `.mid` adds its parts
        beside the ones already here rather than replacing them.
        """
        seen = []
        for track in self.tracks:
            if track.source_midi and track.source_midi not in seen:
                seen.append(track.source_midi)
        return seen

    @property
    def origin(self):
        """The file this song started as, or None for one drawn from nothing.

        The FIRST file, not the only one. It names the song for the map's own
        switch and for the window's title, both of which need one name and
        neither of which can invent one.
        """
        return self.source_midis[0] if self.source_midis else None

    @property
    def notes(self) -> list:
        """Every note in the song, track by track.

        Order is deliberately not the order they are heard in. Nothing
        downstream depends on it: the compiler and the preview both re-sort
        deterministically before anything is written or scheduled.
        """
        return [note for track in self.tracks for note in track.notes]

    def track_by_id(self, track_id: str):
        return next((track for track in self.tracks if track.id == track_id), None)

    def track_for_part(self, source_track: int, channel: int):
        return next(
            (t for t in self.tracks if t.source_track == source_track and t.channel == channel),
            None,
        )

    def new_note_id(self) -> str:
        """An id no note in this song has ever held.

        Prefixed so it can never collide with an imported `channel:pitch:
        occurrence` id, and counted rather than derived from the note itself:
        a derived id is a position, and this whole model exists because
        positions move.
        """
        self.note_serial += 1
        return "n:%d" % self.note_serial

    def new_track_id(self) -> str:
        existing = {track.id for track in self.tracks}
        index = len(self.tracks) + 1
        while ("t:%d" % index) in existing:
            index += 1
        return "t:%d" % index

    def new_drawn_channel(self) -> int:
        """A channel number no hand-drawn track has ever held, including a
        deleted one. See `drawn_channel_serial`'s own docstring for why this
        has to be a counter rather than `max(existing channels) + 1`."""
        self.drawn_channel_serial += 1
        return self.drawn_channel_serial


# ---- JSON round-trip ----


def _note_dict(note: Note) -> dict:
    return {f.name: getattr(note, f.name) for f in fields(Note)}


def _track_dict(track: Track) -> dict:
    payload = {}
    for f in fields(Track):
        value = getattr(track, f.name)
        if f.name == "notes":
            payload["notes"] = [_note_dict(note) for note in value]
        elif isinstance(value, tuple):
            payload[f.name] = list(value)
        else:
            payload[f.name] = value
    return payload


def to_dict(song: Song) -> dict:
    """The song as plain JSON types, in field declaration order.

    Declaration order rather than sorted, so a project file diffs as a change
    somebody made rather than as however a dict happened to be built.
    """
    return {
        "version": song.version,
        "duration_ms": song.duration_ms,
        "loop_start_ms": song.loop_start_ms,
        "loop_end_ms": song.loop_end_ms,
        "loop_enabled": song.loop_enabled,
        "timing": copy.deepcopy(song.timing),
        "conversion": copy.deepcopy(song.conversion),
        "note_serial": song.note_serial,
        "drawn_channel_serial": song.drawn_channel_serial,
        "tracks": [_track_dict(track) for track in song.tracks],
    }


class SongError(ValueError):
    """A project file that cannot be read, with the reason in the message."""


def from_dict(payload) -> Song:
    """Rebuild a song from `to_dict`'s output.

    A version this build does not know is refused outright. Reading the keys
    that happen to still parse is how a document from a later build silently
    loses whatever that build added -- and here that would be note data, not a
    preference.
    """
    if not isinstance(payload, dict):
        raise SongError("a project file has to be a JSON object, got %r" % (payload,))
    version = payload.get("version")
    if version != SONG_VERSION:
        raise SongError(
            "this project says version %r; this build reads and writes version %d only"
            % (version, SONG_VERSION)
        )
    conversion = copy.deepcopy(DEFAULT_CONVERSION)
    conversion.update(payload.get("conversion") or {})
    song = Song(
        version=SONG_VERSION,
        duration_ms=int(payload.get("duration_ms") or 0),
        loop_start_ms=int(payload.get("loop_start_ms") or 0),
        loop_end_ms=int(payload.get("loop_end_ms") or 0),
        loop_enabled=bool(payload.get("loop_enabled", False)),
        timing=copy.deepcopy(payload.get("timing") or {}),
        conversion=conversion,
        note_serial=int(payload.get("note_serial") or 0),
        # Missing in every project written before hand-drawn tracks existed --
        # -1 is exactly what a song that has never minted one already starts
        # at, so an old file reads as "nothing drawn yet" rather than failing.
        # `or 0` is not safe here the way it is for `note_serial` above: 0 is
        # a real, already-used value for this field the moment one track has
        # been drawn, not an absent one.
        drawn_channel_serial=(
            int(payload["drawn_channel_serial"])
            if payload.get("drawn_channel_serial") is not None
            else -1
        ),
    )
    note_names = {f.name for f in fields(Note)}
    track_names = {f.name for f in fields(Track)}
    for raw_track in payload.get("tracks") or []:
        if not isinstance(raw_track, dict):
            raise SongError("a track has to be a JSON object, got %r" % (raw_track,))
        unknown = sorted(set(raw_track) - track_names)
        if unknown:
            raise SongError(
                "track %r: unknown field(s): %s" % (raw_track.get("id"), ", ".join(unknown))
            )
        track = Track(**{k: v for k, v in raw_track.items() if k != "notes"})
        for raw_note in raw_track.get("notes") or []:
            if not isinstance(raw_note, dict):
                raise SongError("a note has to be a JSON object, got %r" % (raw_note,))
            unknown = sorted(set(raw_note) - note_names)
            if unknown:
                raise SongError(
                    "note %r: unknown field(s): %s" % (raw_note.get("id"), ", ".join(unknown))
                )
            track.notes.append(Note(**raw_note))
        song.tracks.append(track)
    return song


# ---- loop export ----


def loop_window(song: Song, start_ms: int, end_ms: int) -> Song:
    """A throwaway copy of `song`, windowed to `[start_ms, end_ms)` and rebased to 0.

    Pure: `song` itself is never touched, which is what lets `Session.export_loop`
    feed this straight into the ordinary compile core without disturbing the
    song the workstation has open. Every track and every lever comes along
    unchanged -- only note membership and timing are windowed -- so the loop
    exports with the same instruments, mutes and tuning the full song has.

    1. A note is kept only if it overlaps the window at all.
    2. A note still sounding at either edge is clipped to it, so the export
       never carries a fragment hanging over from outside the loop.
    3. Every surviving note is rebased by `start_ms`, so the exported timeline
       starts at zero the way any other song does.
    4. The copy's own length becomes exactly the window's, and its own brace
       spans that whole new length with looping off -- a loop exported as a
       song is just a song, not a song with another loop nested inside it.
    """
    windowed = copy.deepcopy(song)
    windowed.duration_ms = end_ms - start_ms
    windowed.loop_start_ms = 0
    windowed.loop_end_ms = windowed.duration_ms
    windowed.loop_enabled = False
    for track in windowed.tracks:
        kept = []
        for note in track.notes:
            note_end = note.start_ms + note.duration_ms
            if note.start_ms >= end_ms or note_end <= start_ms:
                continue
            clipped_start = max(note.start_ms, start_ms)
            clipped_end = min(note_end, end_ms)
            note.start_ms = clipped_start - start_ms
            note.duration_ms = max(1, clipped_end - clipped_start)
            kept.append(note)
        track.notes = kept
    return windowed
