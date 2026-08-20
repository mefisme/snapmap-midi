"""The project: a song, the settings document the window edits it through, and
the file that keeps both.

Two records describe one thing and neither can be dropped. The `Song` holds the
music -- tracks, notes, and every lever resolved onto the track it governs --
and it is what an editor mutates. The settings document holds the same choices
in the shape the window's controls patch and the command line replays, and it is
what a sidecar beside a `.mid` has always been. This module is the one place
that translates between them, because it is the one place allowed to see both: a
`music/` module may not import `settings`, and that layering is what keeps the
MIDI subsystem usable on its own.

Saving a project never touches the `.mid`. An import is a starting point, and a
starting point that gets overwritten is not one -- so the project is its own
file, and the file it came from stays exactly as the composer left it.
"""

from __future__ import annotations

import copy
from pathlib import Path

from snapmap_midi import settings as settings_module
from snapmap_midi.music import importer
from snapmap_midi.music import levers as levers_module
from snapmap_midi.music.midi import for_part
from snapmap_midi.music.song import Song, SongError, from_dict, to_dict
from snapmap_midi.rawmap import codec

#: Appended to the whole name rather than replacing the extension, for the same
#: reason the settings sidecar is: `bach.mid` and `bach.midi` are two different
#: songs and would otherwise share one project file.
PROJECT_SUFFIX = ".smsong.json"

#: Levers read straight off the resolved conversion keywords, as
#: `{track field: (keyword, default)}`. Resolution goes through `for_part` so a
#: named part beats a bare channel exactly the way a conversion would read it.
_RESOLVED = {
    "family": ("channel_families", None),
    "sound": ("channel_sounds", None),
    "percussion": ("part_percussion", "auto"),
    "muted": ("channel_mutes", False),
    "soloed": ("channel_solos", False),
    "volume_db": ("part_volume_db", 0),
    "release_s": ("part_release_s", None),
    "hard_stop": ("part_hard_stop", None),
    "note_off": ("part_note_off", None),
    "note_off_floor_ms": ("part_note_off_floor_ms", None),
    "voices": ("part_voices", None),
    "polyphony": ("part_polyphony", None),
    "sustain_ms": ("part_sustain_ms", None),
    "pitch_octave": ("part_pitch_octave", None),
    "glide_ms": ("part_glide_ms", None),
    "attack_ms": ("part_attack_ms", None),
    "key_range": ("part_key_range", None),
}

#: Choices the window keeps but no conversion reads. They have no lever mapping
#: to resolve through, so they are taken from the settings record itself, with a
#: named part still beating a bare channel.
_RECORD_ONLY = ("pitch_follow_preference", "detected_root_midi")

#: Document keys that describe the user's setup rather than the song, and so
#: survive a project being loaded over the top of a session.
_SETUP_KEYS = ("button", "out_dir", "baseline")


def project_path(midi) -> Path:
    """Where a song's project file lives: beside the song, named after it."""
    return Path(str(midi) + PROJECT_SUFFIX)


def open_midi(mid_path, doc=None, *, mid=None) -> Song:
    """Import a `.mid` and put a settings document's choices onto it.

    This is the migration path for anyone already part way through tuning a
    song: the sidecar beside their file is a validated settings document, and
    every lever in it lands on the track or the note it was written for. Nothing
    is dropped and nothing changes value, so the first compile after the upgrade
    produces the same bytes as the last compile before it.
    """
    song = importer.import_song(mid_path, mid=mid)
    if doc is not None:
        apply_settings(song, doc)
    return song


def apply_settings(song: Song, doc) -> None:
    """Project a validated settings document onto the song, in place.

    A settings document keys its per-channel levers either by a named part or by
    a bare channel number meaning "every part on this channel". A song has no
    such wildcard: its tracks are independently editable, so there is no shared
    identity left for one to hang off. Each wildcard is therefore RESOLVED here,
    onto every track it covered, through the same `for_part` the conversion
    itself uses -- so the per-track values are exactly what the wildcard
    resolved to. What does not carry forward is the LINK: editing one track no
    longer moves its neighbours. That was never a requested behaviour and it
    cannot survive tracks being editable one at a time.

    Every lever is cleared first. A lever the user has just switched off has no
    entry in the document at all, so anything left standing from the previous
    projection would survive being removed -- the control would go back to its
    default and the compile would go on obeying the old value.
    """
    resolved = settings_module.to_compile_kwargs(doc)
    channels = doc["channels"]
    song.conversion = levers_module.conversion_from_tuning(
        doc["tuning"], doc["drums"], doc["drum_keys"]
    )
    for track in song.tracks:
        track.clear_levers()
        source_track, channel = track.part
        for field_name, (keyword, default) in _RESOLVED.items():
            value = for_part(resolved[keyword], source_track, channel, default)
            if field_name == "key_range" and value is not None:
                value = list(value)
            setattr(track, field_name, value)
        if track.sound is not None:
            # The pitch plan resolves as one record rather than field by field,
            # because that is how the conversion reads it: a part that names a
            # sound takes that sound's whole reference, not half of one part's
            # and half of another's.
            profile = for_part(resolved["channel_pitch_profiles"], source_track, channel, {})
            track.pitch_follow = bool(profile.get("pitch_follow", False))
            track.root_midi = profile.get("root_midi")
            track.root_confidence = profile.get("root_confidence")
            track.root_source = profile.get("root_source")
            track.pitch_transpose = float(profile.get("pitch_transpose", 0.0))
            track.fine_tune_cents = float(profile.get("fine_tune_cents", 0.0))
        else:
            track.pitch_transpose = float(
                for_part(resolved["part_transpose"], source_track, channel, 0.0)
            )
        entry = channels.get(track.key) or channels.get(str(channel)) or {}
        for field_name in _RECORD_ONLY:
            if field_name in entry:
                setattr(track, field_name, entry[field_name])
        for note in track.notes:
            override = doc["notes"].get(note.id, {})
            note.pitch_offset = float(override.get("pitch_offset", 0.0))
            note.pitch_semitones = override.get("pitch_semitones")
            note.follow_pitch_semitones = override.get("follow_pitch_semitones")
            note.volume_db = override.get("volume_db")
            note.volume_trim_db = int(override.get("volume_trim_db", 0))


def to_settings(song: Song, base=None) -> dict:
    """The song's choices written back as a validated settings document.

    The window still edits a settings document -- every control it has patches
    one -- so opening a saved project has to hand it one that says what the
    project says. `base` supplies the keys that are about the user's setup
    rather than the song: where maps are written, which baseline to add to, what
    the switch is called.

    Wildcards do not come back. Each track already carries the value its
    wildcard resolved to, so this names every part explicitly -- the same
    conversion, spelled out.
    """
    channels = {}
    for track in song.tracks:
        entry = {
            "family": track.family,
            "percussion": track.percussion,
            "muted": track.muted,
            "soloed": track.soloed,
        }
        if track.volume_db:
            entry["volume_db"] = track.volume_db
        for name in (
            "voices",
            "polyphony",
            "sustain_ms",
            "release_s",
            "hard_stop",
            "note_off",
            "note_off_floor_ms",
            "attack_ms",
            "key_range",
            "pitch_octave",
        ):
            value = getattr(track, name)
            if value is not None:
                entry[name] = value
        if track.pitch_follow_preference is not None:
            entry["pitch_follow_preference"] = track.pitch_follow_preference
        if track.pitch_transpose:
            entry["pitch_transpose"] = track.pitch_transpose
        if track.sound is not None:
            entry["sound"] = track.sound
            entry["pitch_follow"] = track.pitch_follow
            if track.fine_tune_cents:
                entry["fine_tune_cents"] = track.fine_tune_cents
            if track.glide_ms:
                entry["glide_ms"] = track.glide_ms
            if track.root_midi is not None:
                entry["root_midi"] = track.root_midi
                if track.detected_root_midi is not None:
                    entry["detected_root_midi"] = track.detected_root_midi
                if track.root_confidence is not None:
                    entry["root_confidence"] = track.root_confidence
                if track.root_source is not None:
                    entry["root_source"] = track.root_source
        channels[track.key] = entry

    conversion = copy.deepcopy(song.conversion)
    drums = conversion.pop("drums")
    drum_keys = conversion.pop("drum_keys")
    doc = settings_module.defaults(song.origin)
    for key in _SETUP_KEYS:
        if base is not None and base.get(key) is not None:
            doc[key] = base[key]
    doc.update(
        {
            "channels": channels,
            "drums": drums,
            "drum_keys": drum_keys,
            "notes": levers_module.note_overrides(song),
            "tuning": conversion,
        }
    )
    return settings_module.validate(doc)


def save(song: Song, path) -> Path:
    """Write the project. Never the `.mid` -- that is what non-destructive means."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(codec.serialize(to_dict(song)))
    return path


def load(path) -> Song:
    """Read a project, or say what is wrong with the file.

    Every failure arrives as `SongError` naming the path, including a file that
    is not there. The caller is a window showing one message, and it has nowhere
    useful to put a traceback about a file the user chose.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SongError("could not read the project file %s (%s)" % (path, exc)) from exc
    try:
        payload = codec.deserialize(raw)
    except ValueError as exc:
        raise SongError("the project file %s is not valid JSON (%s)" % (path, exc)) from exc
    try:
        return from_dict(payload)
    except SongError as exc:
        raise SongError("%s: %s" % (path, exc)) from exc
