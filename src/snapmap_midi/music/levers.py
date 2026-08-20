"""A song's stored choices as the keyword arguments a conversion reads.

`settings.to_compile_kwargs` does this for a settings document. This does it for
a `Song`, and the two have to agree to the byte: the workstation compiles
through this one and the command line compiles through that one, and a
difference between them is the window exporting something other than what it
drew.

The difference in shape is deliberate. A settings document may key a lever by a
bare channel number, meaning "every part on this channel"; a song has no such
key, because its tracks are independently editable and a wildcard has nothing
left to hang off. Import resolves each wildcard onto the tracks it covered (see
`importer.apply_settings`), so what comes out of here is always a named part.
`for_part` resolves both spellings identically, so nothing downstream can tell.
"""

from __future__ import annotations

import copy

from snapmap_midi.music.song import Song

#: Track lever -> the conversion keyword it feeds, for the levers that are just
#: a value moved across. `None` means "this track defers to the song", so those
#: entries are omitted rather than sent as a null the compiler would read as a
#: setting.
_OPTIONAL_PART_LEVERS = {
    "release_s": "part_release_s",
    "hard_stop": "part_hard_stop",
    "note_off": "part_note_off",
    "note_off_floor_ms": "part_note_off_floor_ms",
    "voices": "part_voices",
    "polyphony": "part_polyphony",
    "sustain_ms": "part_sustain_ms",
    "pitch_octave": "part_pitch_octave",
    "glide_ms": "part_glide_ms",
    "attack_ms": "part_attack_ms",
}


def pitch_profile(track) -> dict:
    """One track's pitch plan for its hand-picked sound.

    Only a track playing an exact sound has one: an automatic instrument is a
    set of pre-tuned recordings, so it reaches for a different recording rather
    than retuning one, and there is no root for it to follow.
    """
    profile = {
        "pitch_follow": track.pitch_follow,
        "root_midi": track.root_midi,
        "root_confidence": track.root_confidence,
        "root_source": track.root_source,
    }
    if track.pitch_transpose:
        profile["pitch_transpose"] = track.pitch_transpose
    if track.fine_tune_cents:
        profile["fine_tune_cents"] = track.fine_tune_cents
    return profile


def note_overrides(song: Song) -> dict:
    """Every per-note expression choice, keyed by that note's own id.

    Keyed rather than carried on the note itself only because that is the shape
    the resolver takes -- and it is a name now, not a position: the id was
    assigned when the note came into existence and moving the note does not
    change it.
    """
    overrides = {}
    for track in song.tracks:
        for note in track.notes:
            entry = {}
            if note.pitch_offset:
                entry["pitch_offset"] = note.pitch_offset
            if note.pitch_semitones is not None:
                entry["pitch_semitones"] = note.pitch_semitones
            if note.follow_pitch_semitones is not None:
                entry["follow_pitch_semitones"] = note.follow_pitch_semitones
            if note.volume_db is not None:
                entry["volume_db"] = note.volume_db
            if note.volume_trim_db:
                entry["volume_trim_db"] = note.volume_trim_db
            if entry:
                overrides[note.id] = entry
    return overrides


def compile_levers(song: Song) -> dict:
    """The song as keyword arguments for a conversion.

    Only arguments `compile_to_rawmap` actually takes: they are splatted into
    the call, so a key it has never heard of is a `TypeError` at export time,
    after the window has already said it is working.
    """
    conversion = song.conversion
    levers = {
        "drums": {"auto": "auto", "on": True, "off": False}[conversion["drums"]],
        "master_volume_db": conversion["master_volume_db"],
        "max_speakers": conversion["max_speakers"],
        "song_polyphony": conversion["song_polyphony"],
        "release_s": conversion["release_s"],
        "hard_stop": conversion["hard_stop"],
        "note_off": conversion["note_off"],
        "note_off_floor_ms": conversion["note_off_floor_ms"],
        "cap_sustain_ms": conversion["cap_sustain_ms"],
        "bass_pitch": conversion["bass_pitch"],
        "bass_cap_ms": conversion["bass_cap_ms"],
        "max_poly": conversion["max_poly"],
        "decaying_families": set(conversion["decaying_families"]),
        "family_caps": dict(conversion["family_caps"]),
        "playback_speed": conversion["playback_speed"],
        "drum_key_overrides": {int(key): sound for key, sound in conversion["drum_keys"].items()},
        "note_overrides": note_overrides(song),
    }
    for keyword in _OPTIONAL_PART_LEVERS.values():
        levers[keyword] = {}
    levers.update(
        {
            "part_volume_db": {},
            "channel_families": {},
            "channel_sounds": {},
            "channel_pitch_profiles": {},
            "part_percussion": {},
            # Mappings, not sets: only a mapping can say that one named part is
            # NOT muted, which a set of muted parts cannot express at all.
            "channel_mutes": {},
            "channel_solos": {},
            "part_transpose": {},
            "part_key_range": {},
        }
    )
    for track in song.tracks:
        part = track.part
        for field_name, keyword in _OPTIONAL_PART_LEVERS.items():
            value = getattr(track, field_name)
            if value is not None:
                levers[keyword][part] = value
        levers["channel_mutes"][part] = track.muted
        levers["channel_solos"][part] = track.soloed
        if track.volume_db:
            levers["part_volume_db"][part] = track.volume_db
        if track.family is not None:
            levers["channel_families"][part] = track.family
        if track.sound is not None:
            levers["channel_sounds"][part] = track.sound
            levers["channel_pitch_profiles"][part] = pitch_profile(track)
        if track.percussion != "auto":
            levers["part_percussion"][part] = track.percussion
        # Every part's transpose, not just the hand-picked-sample ones. An
        # exact sound reads its transpose from the pitch profile above and
        # applies it as a playback modifier on the one recording it has; an
        # automatic instrument reads it from here and applies it by selecting a
        # different pre-tuned recording. Same stored number, two mechanisms,
        # because the two instrument kinds are genuinely different things.
        if track.pitch_transpose:
            levers["part_transpose"][part] = track.pitch_transpose
        if track.key_range is not None:
            levers["part_key_range"][part] = tuple(track.key_range)
    return levers


def conversion_from_tuning(tuning: dict, drums: str, drum_keys: dict) -> dict:
    """A settings document's global section as a song's `conversion`."""
    conversion = copy.deepcopy(tuning)
    conversion["drums"] = drums
    conversion["drum_keys"] = dict(drum_keys)
    return conversion
