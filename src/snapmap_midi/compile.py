"""Orchestration: a MIDI file becomes a playable map.

Scheduling follows expression and decay independently:

  - Neutral decaying notes layer polyphonically on the shared Timeline entity.
  - Decaying notes with pitch or gain use isolated, duration-reserved Timeline emitters
    and are left to fade on their own.
  - Sustained notes use isolated Timeline emitters plus explicit stops or releases.

Voice preparation is shared with browser preview. Per-track voice caps keep an
instrument from monopolising the song, then a global pool enforces the map's
total emitter budget. Density controls reduce exposure to the engine's
emitter-recycling limit; per-note pitch and volume controls describe expression
and do not change that limit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from snapmap_midi.music import importer, pipeline
from snapmap_midi.music.midi import for_part
from snapmap_midi.music.song import Song
from snapmap_midi.rawmap.codec import serialize
from snapmap_midi.rawmap.document import SnapMapDocument
from snapmap_midi.rawmap.palette_refs import PRODUCT_PALETTE_REFS
from snapmap_midi.rawmap.template import blank_map, timeline_position_after_interactive
from snapmap_midi.sound import events as _events
from snapmap_midi.sound.timeline import add_button, ensure_timeline

#: The largest timeline the SnapMap editor can OPEN, in serialized bytes.
#:
#: The engine serializes one entity into a fixed buffer and reports only a byte
#: count, so a timeline past that buffer comes back as a plain failure: the
#: editor says it cannot open the timeline and nothing says why. The map still
#: LOADS and the music still PLAYS -- only editing is lost -- which is what made
#: this so hard to place.
#:
#: Bisected in game over eight loads, measuring the timeline entity through this
#: module's own serializer, so the numbers below compare directly with
#: `stats["timeline_bytes"]`:
#:
#:     1,081,338 bytes  opens
#:     1,091,628 bytes  refuses
#:
#: That is 1.036x one MiB, consistent with a 1 MiB buffer on the engine's side
#: and our compact JSON running a few percent fatter than what it writes. The
#: budget is set at one MiB rather than at the measured line: it is below every
#: size proven to open, it is the round number the evidence points at, and it
#: leaves ~32 KB for the difference between their serializer and ours.
#:
#: Raising a buffer on the READING side cannot help. That was tried first --
#: 1 MB through 32 MB, every one returning zero.
TIMELINE_SERIALIZE_BUDGET = 1024 * 1024

# Automatic Timeline sharding is intentionally parked for now.  The retained
# implementation below can be re-enabled after the listener-fanout design has
# contributor approval and in-game acceptance coverage.  Keeping this as an
# explicit gate, rather than deleting or commenting out the implementation,
# leaves the single-Timeline production behavior unambiguous and the dormant
# path executable in tests.
ENABLE_TIMELINE_SHARDING = False


def installed_event_is_looping(name: str):
    """Loop metadata for an exact installed Play event, when available.

    Kept in the orchestration layer so the independently usable MIDI parser
    does not import upward into the optional installed-game audio subsystem.
    """
    try:
        from snapmap_midi.audio import library

        return library.event_is_looping(name)
    except Exception:
        return None


def installed_event_duration_ms(name: str):
    """Installed non-looping event duration, when soundbank metadata has one."""
    try:
        from snapmap_midi.audio import library

        return library.event_duration_ms(name)
    except Exception:
        return None


def _timeline_self_ref(entity: dict) -> str:
    """The entity reference a Timeline's own empty event group carries."""
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _set_single_timeline_group(entity: dict, target_id: str, events) -> None:
    """Make one Timeline schedule one event list against ``target_id``."""
    entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = {
        "item[0]": {"entity": target_id, "events": _events.events_block(events)},
        "num": 1,
    }


def _layout_song_timelines(doc: SnapMapDocument, switch_uid: int, timelines) -> None:
    """Put the master and auxiliary Unknowns after the song interactive."""

    switch = doc.find_entity(switch_uid)
    spawn = switch["entityDef"]["state"]["edit"].get("spawnPosition", {}) or {}
    anchor = (spawn.get("x", 0.0), spawn.get("y", 0.0), spawn.get("z", 0.0))
    seen = set()
    ordered = []
    for entity in timelines:
        uid = entity["uniqueId"]
        if uid not in seen:
            seen.add(uid)
            ordered.append(entity)
    for index, entity in enumerate(ordered):
        x, y, z = timeline_position_after_interactive(anchor, index)
        entity["entityDef"]["state"]["edit"]["spawnPosition"] = {
            "x": x,
            "y": y,
            "z": z,
        }


def _event_time_batches(events):
    """Adjacent equal-time events, kept atomic when a Timeline is sharded.

    Modifier-before-start order is load-bearing. Splitting two events at the
    same timestamp into separately triggered Timelines would hand their order
    back to the engine scheduler, so every equal-time batch stays together.
    """
    batches = []
    for event in events:
        if not batches or batches[-1][0]["eventTime"] != event["eventTime"]:
            batches.append([])
        batches[-1].append(event)
    return batches


def _write_timeline_shards(
    doc: SnapMapDocument,
    primary: dict,
    target_id: str,
    events,
    display_name: str,
):
    """Store events across editor-safe Timeline entities.

    The first shard is the existing sound emitter itself. Additional entities
    are schedulers targeting that same emitter, so sharding changes only where
    event data is stored; it does not create another audible voice or break
    pitch state. Returns ``[(entity, trigger_reference), ...]``.
    """
    batches = _event_time_batches(events)
    shards = []
    entity = primary
    trigger_ref = _timeline_self_ref(entity)
    chunk = []
    shard_index = 0

    if not batches:
        _set_single_timeline_group(entity, target_id, [])
        return [(entity, trigger_ref)]

    for batch in batches:
        candidate = chunk + batch
        _set_single_timeline_group(entity, target_id, candidate)
        if chunk and len(serialize(entity)) > TIMELINE_SERIALIZE_BUDGET:
            _set_single_timeline_group(entity, target_id, chunk)
            shards.append((entity, trigger_ref))
            shard_index += 1
            entity = doc.add_timeline()
            entity["displayName"] = "%s-s%d" % (display_name, shard_index)
            trigger_ref = _timeline_self_ref(entity)
            chunk = list(batch)
            _set_single_timeline_group(entity, target_id, chunk)
        else:
            chunk = candidate
    shards.append((entity, trigger_ref))
    return shards


#: Arguments to `compile_to_rawmap` that are not conversion levers, and so must
#: not be swept into the levers dict below. `button_name` is here because it is
#: accepted for API compatibility and then ignored: a MIDI export always labels
#: its interactive with the song's own filename, so a probe or a stale sidecar
#: name cannot leak into a later song.
_NOT_A_LEVER = frozenset({"mid_path", "baseline_bytes", "mid", "note_index", "button_name"})


def compile_to_rawmap(
    mid_path,
    baseline_bytes: Optional[bytes] = None,
    button_name: str = "snapmap-midi-song",
    family_overrides: Optional[dict] = None,
    drums="auto",
    max_speakers: int = 32,
    release_s: float = 0.1,
    hard_stop: bool = False,
    note_off: bool = False,
    note_off_floor_ms: Optional[int] = None,
    max_events: Optional[int] = None,
    drop_sustain_over_ms: Optional[int] = None,
    min_sustain_ms: Optional[int] = None,
    drop_shaders: Optional[set] = None,
    cap_sustain_ms: Optional[int] = None,
    bass_pitch: int = 78,
    bass_cap_ms: Optional[int] = None,
    max_poly: Optional[int] = None,
    song_polyphony: int = 32,
    part_voices: Optional[dict] = None,
    part_polyphony: Optional[dict] = None,
    part_glide_ms: Optional[dict] = None,
    part_attack_ms: Optional[dict] = None,
    part_release_s: Optional[dict] = None,
    part_hard_stop: Optional[dict] = None,
    part_note_off: Optional[dict] = None,
    part_note_off_floor_ms: Optional[dict] = None,
    part_sustain_ms: Optional[dict] = None,
    decaying_families: Optional[set] = None,
    family_caps: Optional[dict] = None,
    channel_families: Optional[dict] = None,
    low_split=None,
    drum_overrides: Optional[dict] = None,
    note_index=None,
    channel_mutes: Optional[set] = None,
    drum_key_overrides: Optional[dict] = None,
    channel_solos: Optional[set] = None,
    channel_sounds: Optional[dict] = None,
    channel_pitch_profiles: Optional[dict] = None,
    part_percussion: Optional[dict] = None,
    note_overrides: Optional[dict] = None,
    part_volume_db: Optional[dict] = None,
    master_volume_db: int = 0,
    part_key_range: Optional[dict] = None,
    part_transpose: Optional[dict] = None,
    part_pitch_octave: Optional[dict] = None,
    playback_speed: float = 1.0,
    mid=None,
):
    """Compile a MIDI file into finished map bytes plus a statistics summary.

    A file in, bytes out, with every choice arriving as a keyword. This is the
    command line's entry point and the library's, and its behaviour is fixed:
    the same file and the same levers produce the same bytes they always did.

    Underneath, the file is imported into a transient in-memory `Song` and that
    song is compiled. The workstation holds its song open and edits it instead
    of re-reading a file, but both reach the same core, which is what keeps the
    window's export and this call from drifting apart.

    `mid` accepts an already-parsed `mido.MidiFile`, so a caller holding one
    open song does not pay a fresh disk read and parse on every compile.

    `baseline_bytes` is optional. Omit it and the song is staged in a blank room
    authored from nothing; pass a saved map and the song is added to it, reusing
    that map's timeline if it has one.

    Channel mute/solo state and drum-key overrides are handed straight through;
    all are inert when empty, which lets the workstation pass them on every
    compile without moving a byte.
    """
    # Swept out of the signature rather than listed again. A lever added above
    # and forgotten here would read as unset -- the compiler's own default
    # standing in for something the caller explicitly asked for, silently.
    levers = {name: value for name, value in locals().items() if name not in _NOT_A_LEVER}
    song = importer.import_song(mid_path, mid=mid)
    return compile_song(
        song,
        baseline_bytes,
        levers=levers,
        note_index=note_index,
        song_name=Path(mid_path).name,
    )


def compile_song(
    song: Song,
    baseline_bytes: Optional[bytes] = None,
    *,
    levers: Optional[dict] = None,
    note_index=None,
    song_name: Optional[str] = None,
):
    """Compile an open song into finished map bytes plus a statistics summary.

    The song says what the music is; `levers` says what to convert it with, in
    the keywords `compile_to_rawmap` names. The workstation builds those from
    the song's own stored choices (`music/levers.py`); the command line builds
    them from its flags and a settings file.
    """
    levers = dict(levers or {})
    data = blank_map() if baseline_bytes is None else json.loads(baseline_bytes)
    doc = SnapMapDocument(data=data, palette_refs=PRODUCT_PALETTE_REFS)
    timeline = ensure_timeline(doc)
    timeline_id = timeline["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]

    max_speakers = levers.get("max_speakers", 32)
    release_s = levers.get("release_s", 0.1)
    hard_stop = levers.get("hard_stop", False)
    part_release_s = levers.get("part_release_s") or {}
    part_hard_stop = levers.get("part_hard_stop") or {}
    part_attack_ms = levers.get("part_attack_ms") or {}

    notes, stats = pipeline.resolve(
        song,
        levers,
        note_index=note_index,
        event_is_looping=installed_event_is_looping,
    )
    decaying = [n for n in notes if not n.sustained]
    sustained = [n for n in notes if n.sustained]

    if levers.get("drop_sustain_over_ms") is not None:
        sustained = [n for n in sustained if n.duration <= levers["drop_sustain_over_ms"]]
    if levers.get("min_sustain_ms"):
        # Rapid ornaments churn emitter slots harder than anything else.
        sustained = [n for n in sustained if n.duration >= levers["min_sustain_ms"]]
    if levers.get("drop_shaders"):
        decaying = [n for n in decaying if n.shader not in levers["drop_shaders"]]
    decaying.sort(key=lambda n: (n.start, n.chan, n.source_pitch, n.id))
    if levers.get("max_events"):
        decaying = decaying[: levers["max_events"]]

    prepared = pipeline.prepare(
        decaying + sustained, levers, duration_lookup=installed_event_duration_ms
    )
    decaying, sustained = prepared.decaying, prepared.sustained
    shared_decaying = prepared.shared_decaying
    isolated = prepared.isolated
    count = voices_used = peak_voices = prepared.voice_count

    shared_events = sorted(
        (
            _events.start(n.shader, n.start, channel=_events.LAYERED_CHANNEL)
            for n in shared_decaying
        ),
        key=lambda e: e["eventTime"],
    )
    groups = [(timeline_id, shared_events)]
    timeline_by_target = {timeline_id: timeline}

    by_voice = {}
    for n in isolated:
        by_voice.setdefault(n.voice, []).append(n)

    for voice in range(count):
        voice_notes = sorted(by_voice.get(voice, []), key=lambda n: n.start)
        emitter = doc.add_timeline()
        emitter["displayName"] = "snapmap-midi-v%d" % voice
        emitter_id = _timeline_self_ref(emitter)
        timeline_by_target[emitter_id] = emitter

        scheduled = []
        # When the note before this one on this emitter is guaranteed silent,
        # or None if nothing stops it. Recorded as each note's own stop is
        # scheduled below rather than recomputed here, so the two cannot drift.
        previous_silent_at = None
        for i, n in enumerate(voice_notes):
            glide_ms = int(getattr(n, "glide_ms", 0) or 0)
            start_pitch = (
                getattr(n, "glide_from_pitch") if glide_ms else n.pitch_modifier
            )
            # A LOOPING sound is still playing when the next note on this
            # emitter begins, and `fadePitch`/`fadeSound` address the wildcard
            # channel -- "whatever is currently playing". So without clearing
            # the outgoing loop first, this note's pitch lands on the PREVIOUS
            # note's still-sounding loop and bends that instead, while the
            # sound started just below inherits nothing of its own. Every note
            # ends up pitched by its successor, which reads as pitch wandering
            # unpredictably across the phrase.
            #
            # This does not arise for a one-shot: by the time the next note
            # fires its sample has already finished, so nothing intercepts the
            # modifier and the fresh sound inherits the emitter state -- which
            # is why the ordering below was proven correct against one-shots
            # and still failed on loops. Only a genuine sustain this note is
            # taking over from needs the explicit clear.
            #
            # "Still sounding" deliberately includes a release TAIL, not just
            # an overlapping written block: a note ending 50 ms before the next
            # one, with a 100 ms release, is still audibly fading when that
            # next note retunes the emitter. Checking the previous note's real
            # silence time covers the legato case and that short-gap case with
            # the same test. Equality counts as still sounding, because a stop
            # sharing this note's timestamp has no defined order against its
            # modifier.
            previous = voice_notes[i - 1] if i else None
            if (
                previous is not None
                and previous.sustained
                and (previous_silent_at is None or previous_silent_at >= n.start)
            ):
                # One millisecond earlier so it serializes before this note's
                # own modifier, leaving the emitter genuinely idle for it.
                scheduled.append(_events.stop(max(previous.start, n.start - 1)))
            previous_silent_at = None
            attack_ms = int(
                for_part(part_attack_ms, getattr(n, "track", 0), n.chan, 0) or 0
            )
            # The game does not carry a fade state into a subsequently
            # started one-shot. Start on the exact MIDI time, then address the
            # live sound one millisecond later with a mute and one millisecond
            # after that with its gain ramp. That keeps the attack targeted at
            # an existing engine sound without shifting the arrangement.
            sound_start = n.start
            post_start = sound_start + 1
            if attack_ms:
                scheduled.append(_events.start(n.shader, sound_start))
                scheduled.append(_events.fade(post_start, -60.0, 0.0))
                scheduled.append(_events.fade(post_start + 1, n.volume_db, attack_ms / 1000.0))
            else:
                scheduled.append(_events.fade(n.start, n.volume_db, 0.0))
                scheduled.append(_events.start(n.shader, sound_start))
            # The modifier goes one millisecond AFTER the start, never tied to
            # it.
            #
            # Writing both at the same timestamp and relying on array order was
            # a RACE. It usually worked, and when it did not the start won, the
            # note had no modifier of its own -- modifiers being per-event and
            # never inherited -- and it played at the sample's natural pitch.
            # That is the "random notes play raw" fault: measured live at
            # roughly two notes in eleven, and reproduced against this exact
            # ordering by `tools/create_pitch_race_probe.py`, whose group B
            # (this scheme) came back clean over 24 notes where group A did
            # not.
            #
            # Earlier is not an option: a modifier fired before its start
            # addresses whatever is audible at that instant, which is nothing,
            # and no later sound inherits it -- probed at 2 ms ahead, the whole
            # run played raw. One millisecond after, the sound is definitely
            # playing, which is the case doom-re measured directly ("a modifier
            # reaches a sound that is already playing") and the same trick the
            # glide branch below already relies on. The cost is one millisecond
            # of natural pitch at the head of a note, well under anything
            # audible.
            scheduled.append(_events.fade_pitch(post_start, start_pitch))
            if glide_ms:
                # The ramp follows the instantaneous starting pitch above, so
                # it lands a millisecond later and cannot tie with it.
                scheduled.append(
                    _events.fade_pitch(
                        post_start + 1,
                        n.pitch_modifier,
                        glide_ms / 1000.0,
                    )
                )
            following = voice_notes[i + 1] if i + 1 < len(voice_notes) else None
            voice_cap_end = getattr(n, "voice_cap_end", None)
            stop_at = voice_cap_end if voice_cap_end is not None else n.end
            # Decaying sounds end on their own. Sustains, and one-shots with
            # an explicit Sustain Limit, stop only before a gap; a following
            # start on the same emitter is already the correct cutoff for
            # legato or a stolen voice. A track cap can steal a note even when
            # the global allocator chose another emitter, so it gets an
            # explicit hard stop at its virtual cutoff.
            if voice_cap_end is not None and (following is None or following.start > stop_at):
                scheduled.append(_events.stop(stop_at))
                previous_silent_at = stop_at
            elif getattr(n, "sustain_limited", False):
                # A Sustain Limit is a maximum *audible* duration, not the
                # time at which a release merely begins. Finish the release
                # by its deadline, then actually free the voice: `fadeSound`
                # only ramps volume, it does not stop the underlying one-shot
                # from rendering. Left at a fade alone, a Note-Off-capped
                # sample -- whose whole reason for being capped is that its
                # natural ring runs far past `n.end` -- keeps occupying its
                # voice, silently, all the way to that natural ring's real
                # end. Under a whole song of these the allocator's bookkeeping
                # and the engine's real occupancy drift apart badly enough to
                # exhaust real voices and start recycling emitters out from
                # under still-sounding notes. The browser preview cannot show
                # this: it does not model real voice exhaustion, so it plays
                # a fade-only capped note as cleanly as this stop makes it
                # play in game.
                note_release = for_part(
                    part_release_s,
                    getattr(n, "track", 0),
                    n.chan,
                    release_s,
                )
                note_hard_stop = for_part(
                    part_hard_stop,
                    getattr(n, "track", 0),
                    n.chan,
                    hard_stop,
                )
                if note_hard_stop:
                    scheduled.append(_events.stop(n.end))
                else:
                    release_ms = max(0, int(round(float(note_release) * 1000)))
                    release_start = max(n.start, n.end - release_ms)
                    scheduled.append(
                        _events.fade(
                            release_start,
                            -60.0,
                            (n.end - release_start) / 1000.0,
                        )
                    )
                    # The stop lands exactly when the fade above reaches
                    # silence (`release_start + release duration == n.end`),
                    # never before it -- the release still finishes audibly
                    # first, this only reclaims the voice once nothing is
                    # left to click.
                    scheduled.append(_events.stop(n.end))
                previous_silent_at = n.end
            elif n.sustained and (following is None or following.start > n.end):
                note_release = for_part(
                    part_release_s,
                    getattr(n, "track", 0),
                    n.chan,
                    release_s,
                )
                note_hard_stop = for_part(
                    part_hard_stop,
                    getattr(n, "track", 0),
                    n.chan,
                    hard_stop,
                )
                if note_hard_stop:
                    scheduled.append(_events.stop(n.end))
                    previous_silent_at = n.end
                else:
                    scheduled.append(_events.fade(n.end, -60.0, note_release))
                    # Same reasoning as the sustain_limited branch above:
                    # `fadeSound` only lowers volume, it never releases a
                    # genuinely looping sound's emitter. Nothing else is
                    # coming to retrigger this voice (that is what the
                    # `following` check above already ruled out), so a fade
                    # alone leaves it looping, silently, until the engine's
                    # own recycling eventually reclaims it -- which is
                    # exactly the "recycled note rings to the end of its
                    # sample under the next phrase" the sustain-count
                    # warning describes. The stop once the release finishes
                    # actually frees it instead of leaving that to chance.
                    release_ms = max(0, int(round(float(note_release) * 1000)))
                    scheduled.append(_events.stop(n.end + release_ms))
                    previous_silent_at = n.end + release_ms
        groups.append((emitter_id, sorted(scheduled, key=lambda e: e["eventTime"])))

    entity_events = {
        "item[%d]" % i: {"entity": eid, "events": _events.events_block(evs)}
        for i, (eid, evs) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    timeline["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events

    # Production currently retains one master Timeline and one trigger target:
    # fewer targets and one place to inspect the arrangement. The parked branch
    # below can distribute an oversized master after listener fanout receives
    # contributor approval. Speaker entities never helped this limit because
    # all their events still lived inside the master Timeline.
    unsharded_timeline_bytes = len(serialize(timeline))
    activated_timelines = [(timeline, timeline_id)]
    if ENABLE_TIMELINE_SHARDING and unsharded_timeline_bytes > TIMELINE_SERIALIZE_BUDGET:
        activated_timelines = []
        for target_id, scheduled in groups:
            primary = timeline_by_target[target_id]
            display_name = primary.get("displayName") or "snapmap-midi-shared"
            activated_timelines.extend(
                _write_timeline_shards(
                    doc,
                    primary,
                    target_id,
                    scheduled,
                    display_name,
                )
            )

    trigger_ids = [trigger_ref for _entity, trigger_ref in activated_timelines]
    timeline_sizes = [len(serialize(entity)) for entity, _ref in activated_timelines]
    if song_name is None:
        song_name = Path(song.origin).name if song.origin else "snapmap-midi-song"
    switch_uid = add_button(doc, trigger_ids, song_name)
    layout_entities = list(timeline_by_target.values())
    layout_entities.extend(entity for entity, _trigger_ref in activated_timelines)
    _layout_song_timelines(doc, switch_uid, layout_entities)

    stats.update(
        {
            "notes": len(notes),
            "decaying": len(decaying),
            "sustained": len(sustained),
            "voices": voices_used,
            "events": sum(len(e) for _, e in groups),
            "shared_one_shots": len(shared_decaying),
            "expressive_notes": len(sustained) + len(prepared.expressive_decaying),
            "expressive_one_shots": len(prepared.expressive_decaying),
            "expressive_voices": voices_used,
            "long_sustains": sum(1 for n in sustained if n.duration > 1000),
            "peak_voices": peak_voices,
            "max_speakers": max_speakers,
            # The editor buffer is per Timeline entity. ``timeline_bytes``
            # remains the warning-compatible maximum; the total explains why a
            # large map may contain several individually safe shards.
            "timeline_bytes": max(timeline_sizes, default=0),
            "timeline_total_bytes": sum(timeline_sizes),
            "timeline_unsharded_bytes": unsharded_timeline_bytes,
            "timeline_shards": len(activated_timelines),
            "timeline_budget": TIMELINE_SERIALIZE_BUDGET,
            "timeline_sharding_enabled": ENABLE_TIMELINE_SHARDING,
        }
    )
    return serialize(doc.data), stats
