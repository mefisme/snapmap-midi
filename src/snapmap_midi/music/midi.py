"""MIDI parsing: pair note-on with note-off, and decide what each note plays.

A MIDI file streams events; the compiler needs notes with a start and an end.
Pairing is per (channel, pitch) and stacked, because the same pitch can be
retriggered on a channel before the first one ends.

A note that never receives its note-off is held to the end of the song rather
than dropped. That is the honest reading: the note was still sounding when the
file ran out.

Those are two jobs and they are split here, because only the first one is about
MIDI. `pair_notes` turns a file into written notes -- pitch, velocity, start,
end -- and that answer never changes again; it is what an imported `Song` keeps.
`resolve_notes` takes written notes and the user's choices and decides what each
one PLAYS, which changes every time a dropdown moves. `parse_notes` is still the
two of them back to back, for the library callers and the command line that only
ever wanted a file compiled.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

from snapmap_midi.music.expression import annotate, expression_for, octave_fold
from snapmap_midi.music.gm import DRUM_CHANNEL, SUSTAINED, drum_table, gm_to_family
from snapmap_midi.sound import palette


@dataclass
class Note:
    """One paired note.

    Field order is deliberate and load-bearing. Anything derived from this
    record is serialized in declaration order, and the map format preserves
    key order rather than sorting, so reordering these fields changes output
    bytes with no semantic change whatsoever.
    """

    start: int
    end: int
    shader: str
    sustained: bool
    chan: int
    fam: str
    voice: Optional[int] = None

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class SourceNote:
    """One note as it was WRITTEN, before anything decided what it plays.

    The difference from `Note` above is the whole point of the split: this
    record carries a MIDI pitch and a velocity, while `Note` carries a shader
    and a family. Nothing here depends on a family choice, a mute, a sound
    assignment or a tuning lever, so it survives every one of them unchanged --
    which is what lets an imported song be edited instead of re-parsed.

    `program` is the General MIDI voice in force at this note's own start, kept
    per note rather than per part because a file may change program partway
    through a track.
    """

    track: int
    channel: int
    pitch: int
    velocity: int
    start: int
    end: int
    program: int = 0
    id: str = ""


def messages_with_tracks(mid, speed: float = 1.0):
    """mido's own merge, except each message keeps the track it came from.

    `MidiFile.__iter__` merges the tracks and discards which one each message
    came from. That single omission is why two parts written as separate tracks
    collapsed into one whenever they shared a MIDI channel: by the time the
    first note was paired, the only identity left was `msg.channel`.

    This reproduces mido's merge exactly -- each track to absolute ticks, one
    stable sort so ties keep track order, then ticks to seconds against the
    running tempo -- and yields `(track_index, message, elapsed_seconds)`.
    Tempo is applied after the message that changes it, as mido does, so a
    `set_tempo` governs what follows it rather than the gap before it.

    `speed` is a song-wide playback multiplier, not a tempo: it divides every
    elapsed second uniformly, after the MIDI's own tempo map has already been
    applied, so a tempo change mid-song is still honored relative to the rest
    of the song. 1.0 is the file's own authored speed.

    Type 2 files are refused for the same reason mido refuses them: their tracks
    are asynchronous, so there is no shared clock to merge them onto.
    """
    import mido

    if mid.type == 2:
        raise TypeError("can't merge tracks in a type 2 (asynchronous) MIDI file")

    tagged = []
    for index, track in enumerate(mid.tracks):
        now = 0
        for message in track:
            now += message.time
            tagged.append((now, index, message))
    tagged.sort(key=lambda item: item[0])

    tempo = 500_000
    elapsed = 0.0
    previous = 0
    for tick, index, message in tagged:
        elapsed += mido.tick2second(tick - previous, mid.ticks_per_beat, tempo) / speed
        previous = tick
        yield index, message, elapsed
        if message.type == "set_tempo":
            tempo = message.tempo


def for_part(mapping, track, channel, default=None):
    """The most specific setting for one part.

    Every per-channel lever accepts two kinds of key. A `(track, channel)` pair
    names one part and wins. A bare channel number is the wildcard: it applies to
    every part on that channel, which is both what a settings document written
    before parts existed means and what a user who never split a channel expects.

    Keeping the wildcard rather than expanding it is what lets those documents
    load untouched -- there is no list of parts at the moment settings are
    validated, only at the moment a note is resolved, which is here.
    """
    if (track, channel) in mapping:
        return mapping[(track, channel)]
    return mapping.get(channel, default)


def part_note_ranges(mid) -> dict:
    """Every part's written note span, as `{(track, channel): (lowest, highest)}`.

    Read straight off the tracks rather than from the paired notes, because the
    octave fold has to be decided BEFORE the first note resolves and has to be
    the same for every note in the part. A fold chosen per note would move the
    melody between octaves mid-phrase, which is the one outcome worse than
    clamping.
    """

    ranges: dict = {}
    for track_index, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type != "note_on" or msg.velocity <= 0:
                continue
            key = (track_index, msg.channel)
            low, high = ranges.get(key, (msg.note, msg.note))
            ranges[key] = (min(low, msg.note), max(high, msg.note))
    return ranges


def pair_notes(mid, speed: float = 1.0):
    """Turn a file into written notes. Returns `(notes, end_ms, elapsed_s)`.

    Pairing is unconditional: a note is paired whether or not its track is
    muted, whether or not the palette has a sound for it, and whether or not
    any setting would silence it. That is not a nicety -- it is the model. A
    song holds what was written; a mute is a fact about playback, and a parser
    that skipped muted notes would make un-muting impossible without going back
    to the file.

    Order is note-off order, which is the order the notes finish in rather than
    the order they start. Nothing downstream depends on it -- the compiler and
    the preview both re-sort deterministically -- but it is preserved because
    it is what the single-pass parse produced before this was split out.

    `speed` divides every elapsed second uniformly, after the file's own tempo
    map. 1.0 is the file as authored, and it is the only speed an imported song
    stores: the lever is applied when the song is converted, not when it is
    read.
    """
    program_by_part: dict = {}
    program_by_channel: dict = {}
    # (channel, pitch) -> pending starts. Stacked, because the same pitch can be
    # retriggered on a channel before the first one ends.
    active = defaultdict(list)
    paired: list[SourceNote] = []
    occurrences: defaultdict = defaultdict(int)
    elapsed = 0.0

    for track_index, msg, elapsed in messages_with_tracks(mid, speed=speed):
        now = int(elapsed * 1000)
        if msg.type == "program_change":
            program_by_channel[msg.channel] = msg.program
            program_by_part[(track_index, msg.channel)] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            occurrences[(msg.channel, msg.note)] += 1
            active[(msg.channel, msg.note)].append(
                SourceNote(
                    track=track_index,
                    channel=msg.channel,
                    pitch=msg.note,
                    velocity=msg.velocity,
                    start=now,
                    end=now,
                    program=program_by_part.get(
                        (track_index, msg.channel),
                        program_by_channel.get(msg.channel, 0),
                    ),
                    # The id every settings sidecar on disk already uses, so an
                    # import can carry those overrides straight onto the note
                    # they were written for. It is derived HERE and never
                    # again: once a song exists, an id is a name rather than a
                    # position, which is what survives an insertion.
                    id="%d:%d:%d" % (msg.channel, msg.note, occurrences[(msg.channel, msg.note)]),
                )
            )
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            pending = active.get((msg.channel, msg.note))
            if pending:
                note = pending.pop(0)
                note.end = now
                paired.append(note)

    end = int(elapsed * 1000)
    for pending in active.values():
        for note in pending:
            # Still sounding when the file ended; hold it rather than drop it.
            note.end = end
            paired.append(note)
    return paired, end, elapsed


def source_note_ranges(source_notes) -> dict:
    """Every part's written note span, as `{(track, channel): (lowest, highest)}`.

    The octave fold has to be decided BEFORE the first note resolves and has to
    be the same for every note in the part. A fold chosen per note would move
    the melody between octaves mid-phrase, which is the one outcome worse than
    clamping.
    """
    ranges: dict = {}
    for note in source_notes:
        key = (note.track, note.channel)
        low, high = ranges.get(key, (note.pitch, note.pitch))
        ranges[key] = (min(low, note.pitch), max(high, note.pitch))
    return ranges


def written_is_percussion(channel_pitches, drum_map=None) -> bool:
    """Guess whether the percussion channel really carries a kit.

    Same heuristic as `channel_is_percussion`, asked of `(channel, pitch)` pairs
    rather than of a file, so an imported song can answer it with the file long
    gone. A genuine kit uses few distinct keys and most of them are keys we
    recognise.
    """
    drum_map = drum_table() if drum_map is None else drum_map
    total, recognised, pitches = 0, 0, set()
    for channel, pitch in channel_pitches:
        if channel != DRUM_CHANNEL:
            continue
        total += 1
        pitches.add(pitch)
        if pitch in drum_map:
            recognised += 1
    return total > 0 and len(pitches) <= 20 and recognised / total >= 0.6


def in_part(collection, track, channel) -> bool:
    """Whether a per-part switch is on for this part.

    A SET names only the parts a switch is on for, which cannot express "this
    part specifically is off" -- so a named part could never beat a channel-wide
    entry, and un-muting one part of a muted channel was impossible to say. A
    MAPPING of selector to boolean can say it, and `to_compile_kwargs` sends
    one.

    Sets stay accepted: they are the library API's own spelling and the natural
    way to put it when nothing is being overridden.
    """
    if isinstance(collection, Mapping):
        return bool(for_part(collection, track, channel, False))
    return (track, channel) in collection or channel in collection


def _switch_is_used(collection) -> bool:
    """Whether any part has this switch on. Solo is only in force when one is."""
    if isinstance(collection, Mapping):
        return any(collection.values())
    return bool(collection)


def is_percussion_part(modes, track, channel, drums_on: bool) -> bool:
    """Whether this part is a drum kit.

    `auto` is the General MIDI convention -- channel 10, if the heuristic
    accepted it as a kit -- and is right for nearly every file. A part may
    also say outright which it is, because the convention is only a
    convention: a composer may write a kit to channel 6, and before this
    existed those notes were mapped as piano and nothing said so.

    The heuristic is deliberately NOT extended to other channels. Its keys
    span B1 to F5, which is exactly where bass lines sit, so a sparse bass
    part would qualify as a kit and be turned into drums with no warning.
    """
    mode = for_part(modes, track, channel, "auto")
    if mode == "kit":
        return True
    if mode == "melodic":
        return False
    return channel == DRUM_CHANNEL and drums_on


def _record(note):
    """Freeze the note's written length before any scheduling policy touches it.

    `note.end` is the note's SCHEDULED end and later stages are entitled to move
    it: `prepare_voice_layers` shortens it for `cap_sustain_ms` and the family
    caps, and speaker stealing cuts it shorter still. Every one of those is a
    consequence of the tuning levers rather than a fact about the file, so a
    piano roll drawn from `note.end` redraws the composition every time a slider
    moves and the user cannot tell a thinned passage from lost data.

    `midi_end` is the parsed note-off and nothing downstream writes to it. It is
    set here, at the one moment it is known to be untouched.

    Set as an attribute rather than a `Note` field on purpose: field order in
    that dataclass is load-bearing for the exported bytes, and this value is
    never exported.
    """
    note.midi_end = note.end
    return note


def channel_is_percussion(mid, drum_map=None) -> bool:
    """Guess whether the percussion channel really carries a kit.

    Some files put a melodic part on the percussion channel, and some put a
    real kit elsewhere. The heuristic: a genuine kit uses few distinct keys
    and most of them are keys we recognise. The count is over the whole file
    regardless of which track a key came from, so nothing here needs the
    tracks merged into playback order -- and asking for that order is the
    expensive part: `MidiFile.__iter__` recomputes absolute time and copies
    every message to do it, for a question this never asks. Reading each
    track's own messages directly answers the same question without paying
    for an ordering nobody uses.
    """
    # Resolved at call time, not bound as a default: the default would freeze
    # the shipped table at import, and a user who mapped the exotic keys their
    # own kit uses would still be told their kit is not one.
    drum_map = drum_table() if drum_map is None else drum_map
    total, recognised, pitches = 0, 0, set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0 and msg.channel == DRUM_CHANNEL:
                total += 1
                pitches.add(msg.note)
                if msg.note in drum_map:
                    recognised += 1
    return total > 0 and len(pitches) <= 20 and recognised / total >= 0.6


def _exact_sound_sustained(
    shader: str, family: str, no_sustain, event_is_looping, event_looping_cache
) -> bool:
    """Whether an exact assignment needs a paired stop event.

    Palette sounds retain their established family behavior. For a full-game
    event, installed catalog metadata decides. An unknown manually entered
    Play event -- no installed record, or the lookup failed -- is treated as
    a one-shot rather than a loop. The reverse used to be true on the theory
    that failing to stop a real loop leaks an emitter while stopping a
    one-shot that already ended is harmless, but that theory assumed
    "sustained" always got an explicit stop; it did not, until the same
    session that found this. Most hand-picked sounds are one-shots, so
    defaulting the unconfirmed case to looping was routing the common case
    through sustained's musical-hold assumptions (no per-note Note Off cap,
    glide between notes on the same voice) for a sound that cannot actually
    deliver them -- audible as an unwanted retrigger at every note change.
    """
    if family != "exact":
        return (family in SUSTAINED or family.startswith("amb_")) and family not in no_sustain
    if shader not in event_looping_cache:
        try:
            event_looping_cache[shader] = (
                event_is_looping(shader) if event_is_looping is not None else None
            )
        except Exception:
            event_looping_cache[shader] = None
    return event_looping_cache[shader] is True


def resolve_notes(
    source_notes,
    *,
    drums_on: bool = False,
    note_ranges=None,
    duration_s: float = 0.0,
    song_duration_ms: Optional[float] = None,
    drum_defaults=None,
    family_overrides=None,
    decaying_families=None,
    channel_families=None,
    low_split=None,
    drum_overrides=None,
    note_index=None,
    channel_mutes=None,
    drum_key_overrides=None,
    channel_solos=None,
    channel_sounds=None,
    event_is_looping=None,
    channel_pitch_profiles=None,
    note_overrides=None,
    part_volume_db=None,
    master_volume_db=0,
    include_silent=False,
    part_percussion=None,
    part_key_range=None,
    part_transpose=None,
    part_pitch_octave=None,
):
    """Decide what each written note plays, and summarise what that cost.

    Everything above this line is a fact about the file. Everything here is a
    choice the user made, which is why the two are separable at all: this runs
    again for every dropdown, and nothing it does writes back into the written
    notes it was handed.

    Family selection runs in increasing order of specificity: the program's
    default family, then a family-wide override, then a per-channel override,
    then a pitch split that sends the low register somewhere else. The last one
    to apply wins.

    `channel_mutes` silences whole channels. When any channel is in
    `channel_solos`, only soloed, unmuted channels are audible; mute wins when
    both switches are set. Silent notes are not counted in `dropped` because
    they were excluded deliberately rather than lost by the palette. The UI may
    set `include_silent` to retain those notes for a dimmed piano-roll display,
    while preview audio and map export use only notes whose `audible` metadata
    is true.

    `song_duration_ms` is the song's own editable length (`Song.duration_ms`),
    already on whatever clock `source_notes` themselves are -- the caller
    divides by playback speed before this ever sees it, the same as every
    other millisecond value in this function. A note starting at or past it is
    silenced the same way a mute silences one: never deleted, never clipped,
    just excluded from what plays and exported -- so shrinking a song's length
    is exactly as reversible as un-muting a channel. `None` (the default, and
    what every caller outside the workstation passes) means no such boundary
    exists at all.

    `drum_key_overrides` gives a percussion key its sound by MIDI key number,
    which is the only lever that reaches the exotic keys `DRUM_MAP` drops on
    purpose. It is keyed by key rather than by shader precisely so it can name a
    key that currently resolves to nothing at all.

    `note_overrides` is keyed by each written note's own id. Which id that is,
    and what keeps it stable, is the caller's business: a `Song` hands over its
    notes' own permanent ids, and a bare `parse_notes` hands over the derived
    `channel:pitch:occurrence` ids a settings sidecar uses.
    """
    family_overrides = family_overrides or {}
    drum_overrides = drum_overrides or {}
    drum_key_overrides = drum_key_overrides or {}
    channel_families = channel_families or {}
    channel_sounds = channel_sounds or {}
    channel_mutes = channel_mutes or frozenset()
    channel_pitch_profiles = channel_pitch_profiles or {}
    note_overrides = note_overrides or {}
    part_volume_db = part_volume_db or {}
    no_sustain = set(decaying_families or ())
    channel_solos = channel_solos or frozenset()
    part_percussion = part_percussion or {}
    part_key_range = part_key_range or {}
    part_transpose = part_transpose or {}
    part_pitch_octave = part_pitch_octave or {}
    solos_in_force = _switch_is_used(channel_solos)
    # Once, not once per note. This opens the user's percussion table, and the
    # loop below runs for every written note in the song.
    drum_defaults = drum_table() if drum_defaults is None else drum_defaults
    event_looping_cache = {}
    low_cut, low_family = low_split or (0, None)
    sound_categories = palette.sound_categories()

    if note_ranges is None:
        note_ranges = source_note_ranges(source_notes)
    octave_shifts: dict = {}

    def octave_shift_for(track, channel, root, transpose, cents) -> int:
        """This part's octave fold, decided once and reused for every note."""
        key = (track, channel)
        if key not in octave_shifts:
            override = for_part(part_pitch_octave, track, channel, None)
            if override is not None:
                octave_shifts[key] = int(override)
            else:
                low, high = note_ranges.get(key, (None, None))
                octave_shifts[key] = octave_fold(
                    root, low, high, float(transpose) + float(cents) / 100.0
                )
        return octave_shifts[key]

    index = note_index if note_index is not None else palette.build_note_index()

    notes: list[Note] = []
    dropped = 0

    for source in source_notes:
        track_index = source.track
        channel = source.channel
        pitch = source.pitch
        note_id = source.id
        muted = in_part(channel_mutes, track_index, channel)
        solo_excluded = solos_in_force and not in_part(channel_solos, track_index, channel)
        key_range = for_part(part_key_range, track_index, channel, None)
        out_of_key_range = key_range is not None and not (key_range[0] <= pitch <= key_range[1])
        beyond_length = song_duration_ms is not None and source.start >= song_duration_ms
        audible = not muted and not solo_excluded and not out_of_key_range and not beyond_length
        if not audible and not include_silent:
            continue
        override = note_overrides.get(note_id, {})
        pitch_offset = float(override.get("pitch_offset", 0))
        manual_pitch_semitones = (
            float(override["pitch_semitones"]) if "pitch_semitones" in override else None
        )
        follow_pitch_semitones = (
            float(override["follow_pitch_semitones"])
            if "follow_pitch_semitones" in override
            else None
        )
        note_volume_db = int(override["volume_db"]) if "volume_db" in override else None
        volume_trim_db = int(override.get("volume_trim_db", 0))
        track_volume_db = int(for_part(part_volume_db, track_index, channel, 0))
        exact_sound = for_part(channel_sounds, track_index, channel)
        chosen_family = for_part(channel_families, track_index, channel)
        applied_root = None
        profile_root = None
        root_confidence = None
        root_source = None
        pitch_follow = False
        track_transpose = 0.0
        octave_shift = 0
        fine_tune_cents = 0.0
        # An automatic instrument is a SET of separately recorded, pre-tuned
        # samples -- one per key, 88 of them for piano. Transposing it should
        # therefore reach for a different recording rather than retune one,
        # which is both what the real instrument sounds like an octave up and
        # free: the pitch modifier stays 0, so the note keeps its place on the
        # shared unlimited-polyphony emitter instead of claiming a dedicated
        # voice. A hand-picked exact sound has only the one recording, so it
        # keeps reading transpose out of its pitch profile as a playback
        # modifier -- the branches below are mutually exclusive, so no note is
        # ever transposed twice.
        auto_transpose = (
            0
            if exact_sound is not None
            else int(for_part(part_transpose, track_index, channel, 0) or 0)
        )
        # Clamped because it only picks a sample; MIDI has no note above 127
        # and `decl_for` would just fall back to the nearest one anyway.
        sample_note = max(0, min(127, pitch + auto_transpose))
        if exact_sound is not None:
            shader = exact_sound
            family = sound_categories.get(shader, "exact")
            profile = for_part(channel_pitch_profiles, track_index, channel, {})
            profile_root = profile.get("root_midi")
            root_confidence = profile.get("root_confidence")
            root_source = profile.get("root_source")
            track_transpose = float(profile.get("pitch_transpose", 0))
            fine_tune_cents = float(profile.get("fine_tune_cents", 0))
            pitch_follow = bool(profile.get("pitch_follow", False) and profile_root is not None)
            if pitch_follow:
                applied_root = float(profile_root)
                # Only a following sound can be out of reach: it is the one
                # whose modifier grows with the distance between the
                # recording's natural note and the music. A sound playing at
                # its natural pitch asks for 0 no matter where it was
                # recorded.
                octave_shift = octave_shift_for(
                    track_index,
                    channel,
                    applied_root,
                    track_transpose,
                    fine_tune_cents,
                )
            # Full-game exact events take their loop behavior from the
            # installed event catalog. Curated palette assignments preserve
            # the established family scheduling rules.
            sustained = _exact_sound_sustained(
                shader,
                family,
                no_sustain,
                event_is_looping,
                event_looping_cache,
            )
        elif chosen_family is not None:
            # A pitched family selected for channel 10 is still a pitched
            # instrument. The explicit track choice wins over automatic
            # percussion detection, so drums need no separate workspace.
            family = chosen_family
            shader = palette.decl_for(family, sample_note, index)
            sustained = family in SUSTAINED and family not in no_sustain
        elif is_percussion_part(part_percussion, track_index, channel, drums_on):
            # The per-key choice is the user's and is final. `drum_overrides`
            # is keyed by resolved shader and exists to retimbre what the
            # TABLE picked, so applying it after a per-key override would
            # silently replace the sound someone had just chosen.
            shader = drum_key_overrides.get(pitch)
            if shader is None:
                shader = drum_defaults.get(pitch)
                if shader:
                    shader = drum_overrides.get(shader, shader)
            sustained, family = False, "drums"
        else:
            family = gm_to_family(source.program)
            family = family_overrides.get(family, family)
            if low_family and pitch < low_cut:
                family = low_family
            shader = palette.decl_for(family, sample_note, index)
            sustained = family in SUSTAINED and family not in no_sustain
        if shader:
            if exact_sound is None and family != "drums":
                profile_root = palette.shader_pitch(shader)
                if profile_root is not None:
                    # `profile_root` stays the honest natural note of the
                    # recording that was chosen. The root HANDED to the
                    # expression math is offset by the transpose so the
                    # modifier resolves against `sample_note` rather than
                    # the written note: with a recording available at the
                    # transposed pitch that lands on exactly 0 (nothing to
                    # correct, shared path preserved), and where the family
                    # has no sample there it becomes precisely the leftover
                    # correction needed to reach the transposed pitch from
                    # whichever neighbour `decl_for` fell back to.
                    applied_root = float(profile_root) - auto_transpose
                    root_confidence = 1.0
                    root_source = "palette_name"
                    pitch_follow = True
            active_pitch_semitones = (
                follow_pitch_semitones if pitch_follow else manual_pitch_semitones
            )
            expression = expression_for(
                pitch,
                source.velocity,
                applied_root,
                pitch_offset=pitch_offset,
                pitch_semitones=active_pitch_semitones,
                track_transpose=track_transpose,
                octave_shift=octave_shift,
                fine_tune_cents=fine_tune_cents,
                volume_trim_db=volume_trim_db,
                note_volume_db=note_volume_db,
                track_volume_db=track_volume_db,
                master_volume_db=master_volume_db,
            )
            metadata = {
                "id": note_id,
                # Which track wrote this note. Deliberately NOT part of `id`:
                # an imported id is already unique without it, and adding it
                # would invalidate every `note_overrides` entry in every
                # settings sidecar already on disk.
                "track": track_index,
                "profile_root_pitch": profile_root,
                "root_confidence": root_confidence,
                "root_source": root_source,
                "pitch_follow": pitch_follow,
                # Glide is meaningful only when successive notes retune
                # one recording. Automatic instruments deliberately choose
                # separately tuned recordings and keep their shared,
                # polyphonic emitter path instead.
                "uses_exact_sound": exact_sound is not None,
                "manual_pitch_semitones": manual_pitch_semitones,
                "follow_pitch_semitones": follow_pitch_semitones,
                "audible": audible,
                "muted": muted,
                "solo_excluded": solo_excluded,
                "out_of_key_range": out_of_key_range,
                "beyond_length": beyond_length,
            }
            note = Note(source.start, source.end, shader, sustained, channel, family)
            notes.append(_record(annotate(note, expression, **metadata)))
        else:
            if audible:
                dropped += 1

    audible_notes = [note for note in notes if getattr(note, "audible", True)]
    pitch_limits = {}
    for note in audible_notes:
        if not note.pitch_limited:
            continue
        detail = pitch_limits.setdefault(
            note.chan,
            {
                "channel": note.chan,
                "count": 0,
                "source_low": note.source_pitch,
                "source_high": note.source_pitch,
                "requested_low": note.requested_pitch,
                "requested_high": note.requested_pitch,
                "applied_low": note.pitch_modifier,
                "applied_high": note.pitch_modifier,
            },
        )
        detail["count"] += 1
        detail["source_low"] = min(detail["source_low"], note.source_pitch)
        detail["source_high"] = max(detail["source_high"], note.source_pitch)
        detail["requested_low"] = min(detail["requested_low"], note.requested_pitch)
        detail["requested_high"] = max(detail["requested_high"], note.requested_pitch)
        detail["applied_low"] = min(detail["applied_low"], note.pitch_modifier)
        detail["applied_high"] = max(detail["applied_high"], note.pitch_modifier)

    # Reported so the window can SAY that a part was moved. Folding silently
    # would trade one confusing outcome for another: the melody would survive,
    # but nobody could tell why the sound sits an octave from where it was
    # written.
    folded = {}
    for note in audible_notes:
        shift = getattr(note, "octave_shift", 0)
        if shift:
            folded.setdefault(note.chan, shift)

    stats = {
        "drums_on": drums_on,
        "octave_shift_channels": [
            {"channel": channel, "octaves": folded[channel]} for channel in sorted(folded)
        ],
        "dropped": dropped,
        "duration_s": duration_s,
        "pitch_adjusted": sum(note.pitch_modifier != 0 for note in audible_notes),
        "volume_adjusted": sum(note.volume_db != 0 for note in audible_notes),
        "pitch_limited": sum(note.pitch_limited for note in audible_notes),
        "pitch_limit_channels": [pitch_limits[channel] for channel in sorted(pitch_limits)],
        "volume_limited": sum(note.volume_limited for note in audible_notes),
    }
    return notes, stats


def parse_notes(
    mid_path,
    drums="auto",
    family_overrides=None,
    decaying_families=None,
    channel_families=None,
    low_split=None,
    drum_overrides=None,
    note_index=None,
    channel_mutes=None,
    drum_key_overrides=None,
    channel_solos=None,
    channel_sounds=None,
    event_is_looping=None,
    channel_pitch_profiles=None,
    note_overrides=None,
    part_volume_db=None,
    master_volume_db=0,
    include_silent=False,
    part_percussion=None,
    part_key_range=None,
    part_transpose=None,
    part_pitch_octave=None,
    playback_speed: float = 1.0,
    mid=None,
):
    """Parse a MIDI file into paired notes plus a statistics summary.

    Pairing and resolution back to back: this is the library and command-line
    surface, for a caller that has a file and wants notes rather than a song to
    edit. The workstation pairs once, at import, and resolves per change.

    `mid` accepts an already-parsed `mido.MidiFile`, for a caller that reuses
    one parse across several calls against the same file instead of paying disk
    read and message validation again for every one of them. Omit it and this
    parses `mid_path` itself.

    `playback_speed` scales every note's start and end uniformly relative to
    the MIDI's own tempo map; 1.0 plays it exactly as authored.
    """
    import mido

    # clip=True clamps out-of-range data bytes rather than refusing the file.
    if mid is None:
        mid = mido.MidiFile(str(mid_path), clip=True)
    source_notes, _end_ms, elapsed = pair_notes(mid, speed=playback_speed)
    drum_defaults = drum_table()
    drums_on = channel_is_percussion(mid, drum_defaults) if drums == "auto" else bool(drums)
    return resolve_notes(
        source_notes,
        drums_on=drums_on,
        duration_s=round(elapsed, 2),
        drum_defaults=drum_defaults,
        family_overrides=family_overrides,
        decaying_families=decaying_families,
        channel_families=channel_families,
        low_split=low_split,
        drum_overrides=drum_overrides,
        note_index=note_index,
        channel_mutes=channel_mutes,
        drum_key_overrides=drum_key_overrides,
        channel_solos=channel_solos,
        channel_sounds=channel_sounds,
        event_is_looping=event_is_looping,
        channel_pitch_profiles=channel_pitch_profiles,
        note_overrides=note_overrides,
        part_volume_db=part_volume_db,
        master_volume_db=master_volume_db,
        include_silent=include_silent,
        part_percussion=part_percussion,
        part_key_range=part_key_range,
        part_transpose=part_transpose,
        part_pitch_octave=part_pitch_octave,
    )
