"""One conversion, shared by the map export and the window's own preview.

These two used to be two copies of the same eight steps -- polyphony thinning,
per-track voice caps, the global speaker pool, allocation, glide -- written out
separately in `compile.py` and in `Session.preview_manifest`. Two copies of a
policy is two policies: a fix to one of them is a preview that no longer
describes the export, which is the single thing the window exists to prevent.

They were written twice because they consumed different things: one started from
a MIDI path and the other from a settings document. Both now start from a
`Song`, so there is nothing left to keep them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from snapmap_midi.music.gm import drum_table
from snapmap_midi.music.midi import (
    SourceNote,
    for_part,
    resolve_notes,
    written_is_percussion,
)
from snapmap_midi.music.song import Song
from snapmap_midi.music.voices import (
    allocate_voices,
    apply_glides,
    apply_voice_cap,
    prepare_voice_layers,
    thin_global_polyphony,
    thin_polyphony,
    thin_simultaneous,
)

#: Conversion keywords `resolve_notes` reads. Named rather than splatted whole,
#: because the levers dict also carries scheduling and serialization choices
#: that mean nothing until after a note has resolved.
_RESOLVE_KEYS = (
    "family_overrides",
    "decaying_families",
    "channel_families",
    "low_split",
    "drum_overrides",
    "channel_mutes",
    "drum_key_overrides",
    "channel_solos",
    "channel_sounds",
    "channel_pitch_profiles",
    "note_overrides",
    "part_volume_db",
    "master_volume_db",
    "part_percussion",
    "part_key_range",
    "part_transpose",
    "part_pitch_octave",
)


def source_notes(song: Song, speed: float = 1.0) -> list:
    """The song's written notes on the clock the transport will run them at.

    A song stores one clock -- its own, at speed 1.0 -- because `playback_speed`
    is a lever somebody is auditioning rather than a fact about the music. It is
    applied here, on the way out, so moving that slider never rewrites a note.
    """
    out = []
    for track in song.tracks:
        for note in track.notes:
            start, end = note.start_ms, note.end_ms
            if speed != 1.0:
                start = int(start / speed)
                end = int(end / speed)
            out.append(
                SourceNote(
                    track=track.source_track,
                    channel=track.channel,
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start=start,
                    end=end,
                    program=note.program,
                    id=note.id,
                )
            )
    return out


def resolve(
    song: Song,
    levers: dict,
    *,
    note_index=None,
    include_silent=False,
    event_is_looping=None,
):
    """Decide what every note in the song plays. Returns `(notes, stats)`."""
    speed = levers.get("playback_speed", 1.0)
    sources = source_notes(song, speed)
    drums = levers.get("drums", "auto")
    drum_defaults = drum_table()
    drums_on = (
        written_is_percussion(((n.channel, n.pitch) for n in sources), drum_defaults)
        if drums == "auto"
        else bool(drums)
    )
    return resolve_notes(
        sources,
        drums_on=drums_on,
        duration_s=round(song.duration_ms / 1000.0 / speed, 2),
        # Same clock the sources above were just put on: a note starting past
        # the song's own length is excluded from playback and export exactly
        # like a mute, never deleted or clipped (see `resolve_notes`'s own
        # docstring). `0` is treated as "no length recorded yet" rather than
        # "silence everything" -- only a hand-built `Song` nothing has ever
        # imported into can carry it, and it must resolve exactly as before
        # this existed.
        song_duration_ms=(song.duration_ms / speed) if song.duration_ms else None,
        drum_defaults=drum_defaults,
        note_index=note_index,
        include_silent=include_silent,
        event_is_looping=event_is_looping,
        **{key: levers[key] for key in _RESOLVE_KEYS if key in levers},
    )


@dataclass
class Prepared:
    """What the scheduler and the preview both need out of one conversion."""

    decaying: list = field(default_factory=list)
    sustained: list = field(default_factory=list)
    shared_decaying: list = field(default_factory=list)
    expressive_decaying: list = field(default_factory=list)
    isolated: list = field(default_factory=list)
    voice_count: int = 0


def prepare(notes, levers: dict, *, duration_lookup=None) -> Prepared:
    """Thin, cap and allocate, in the one order both consumers must agree on.

    Per-track polyphony first, then the song-wide polyphony limit, then the
    duration policy, then each track's own voice budget, and only then the one
    global speaker pool. The order is the meaning: a track cap that ran after
    the global pool would hand spare speakers back to a track the user had
    already limited.

    Every note that does not survive a step is told which step it was, and every
    note whose end is moved records where playback really stops. Only the window
    reads those -- the exported map is written from the notes that survived, and
    a stolen emitter is silent whether or not anything wrote down why -- but
    they are set here rather than in the preview so the two readings come from
    one pass and cannot disagree.
    """
    parts: dict = {}
    for note in notes:
        parts.setdefault((getattr(note, "track", 0), note.chan), []).append(note)
    limited_notes = []
    for part_key in sorted(parts):
        part = parts[part_key]
        poly = for_part(levers.get("part_polyphony") or {}, *part_key, levers.get("max_poly"))
        kept = thin_polyphony(part, poly) if poly else part
        _blame(part, kept, "polyphony")
        limited_notes.extend(kept)

    globally_kept = thin_global_polyphony(limited_notes, levers.get("song_polyphony", 32))
    _blame(limited_notes, globally_kept, "global_polyphony")
    limited_notes = globally_kept

    decaying = [note for note in limited_notes if not note.sustained]
    sustained = [note for note in limited_notes if note.sustained]

    shared_decaying, expressive_decaying, layers = prepare_voice_layers(
        decaying,
        sustained,
        cap_sustain_ms=levers.get("cap_sustain_ms"),
        bass_pitch=levers.get("bass_pitch", 78),
        bass_cap_ms=levers.get("bass_cap_ms"),
        family_caps=levers.get("family_caps"),
        duration_lookup=duration_lookup,
        part_glide_ms=levers.get("part_glide_ms"),
        part_attack_ms=levers.get("part_attack_ms"),
        part_voices=levers.get("part_voices"),
        part_sustain_ms=levers.get("part_sustain_ms"),
        note_off=levers.get("note_off", False),
        part_note_off=levers.get("part_note_off"),
        note_off_floor_ms=levers.get("note_off_floor_ms"),
        part_note_off_floor_ms=levers.get("part_note_off_floor_ms"),
    )

    max_speakers = levers.get("max_speakers", 32)
    isolated = []
    for part_key in sorted(layers):
        layer = layers[part_key]
        voices = for_part(levers.get("part_voices") or {}, *part_key, max_speakers)
        kept = apply_voice_cap(layer, voices)
        _blame(layer, kept, "voices")
        for note in kept:
            cap_end = getattr(note, "voice_cap_end", None)
            if cap_end is not None:
                note.preview_cut = True
                note.preview_end = cap_end
                if cap_end < note.end:
                    note.shortened_by = "voices"
        isolated.extend(kept)

    # One pool for the whole song: the global voice count decides how many
    # dedicated pitch-controlled emitters the map and preview may use in total.
    kept = thin_simultaneous(isolated, max_speakers)
    _blame(isolated, kept, "voices")
    isolated = kept
    voice_count = allocate_voices(isolated, max_speakers)
    apply_glides(isolated, levers.get("part_glide_ms"))
    _record_steals(isolated)

    return Prepared(
        decaying=decaying,
        sustained=sustained,
        shared_decaying=shared_decaying,
        expressive_decaying=expressive_decaying,
        isolated=isolated,
        voice_count=voice_count,
    )


def _blame(before, kept, limit: str) -> None:
    """Name the lever that silenced each note the step dropped."""
    surviving = {id(note) for note in kept}
    for note in before:
        if id(note) not in surviving:
            note.limited_by = limit
    return None


def _record_steals(notes) -> None:
    """Where each note really stops once the next one takes its emitter.

    Starting a note on a stolen emitter cuts off the note that owned it. The
    exported map gets that for free -- the engine simply stops rendering the
    old sound -- so this exists for the window, which would otherwise draw a
    note ringing through the phrase that silenced it.
    """
    by_voice: dict = {}
    for note in notes:
        by_voice.setdefault(note.voice, []).append(note)
    for voice_notes in by_voice.values():
        voice_notes.sort(key=lambda note: note.start)
        for index, note in enumerate(voice_notes[:-1]):
            following = voice_notes[index + 1]
            effective_end = note.end if note.sustained else getattr(note, "voice_end", note.end)
            effective_end = min(effective_end, getattr(note, "voice_cap_end", effective_end))
            if following.start <= effective_end:
                note.preview_cut = True
            if following.start < effective_end:
                note.shortened_by = "voices"
                note.preview_end = min(getattr(note, "preview_end", effective_end), following.start)
