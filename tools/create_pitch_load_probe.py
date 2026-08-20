"""Second pass on the wandering-pitch fault, testing what probe one did not.

`create_pitch_wander_probe.py` came back entirely clean in game: isolated
pitches, rapid notes on one emitter, simultaneous notes on separate emitters,
and an eight-note round-robin scale were all correct. So the engine handles the
failing song's spacing, its emitter count and its sound.

Three things that song does were still untested, and this probe adds them:

  1. **Note Off capping.** The sidecar sets `note_off` with a 230 ms floor, so
     the real export writes a release `fadeSound` and a `stopSound` for 307 of
     its 385 notes. Probe one wrote neither. Those land on the WILDCARD
     channel -- "whatever is playing" -- which is the one event shape whose
     reach doom-re's probe measured for pitch but never for a stop.
  2. **Fractional pitch.** The song asks for values like -10.24 and +22.76.
     Probe one only used whole semitones.
  3. **Sustained load.** 385 notes across 53 s, against probe one's eight.
     `voices.py` documents that the engine recycles emitter slots under load
     and that a recycled one-shot can no longer be stopped.

Groups 1-3 change one of those at a time against a known-good baseline; group 4
runs all of them together for long enough that a slot-recycling failure has room
to appear.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from snapmap_midi.rawmap.codec import serialize
from snapmap_midi.rawmap.document import SnapMapDocument
from snapmap_midi.rawmap.palette_refs import PRODUCT_PALETTE_REFS
from snapmap_midi.rawmap.template import blank_map
from snapmap_midi.sound.events import (
    START_CHANNEL,
    events_block,
    fade,
    fade_pitch,
    start,
    stop,
)
from snapmap_midi.sound.timeline import add_button, ensure_timeline

SAMPLE = "Play_sfx_stranglerstrings_02"

#: The song's own numbers: notes every 136 ms across four emitters, each note
#: capped at the 230 ms Note Off floor with a 100 ms release before it.
GAP_MS = 136
EMITTERS = 4
CAP_MS = 230
RELEASE_MS = 100

#: Whole semitones -- the baseline probe one already proved correct.
WHOLE_SCALE = (0, 2, 4, 5, 7, 9, 11, 12)

#: Real values taken from the failing export, spanning its true -10.24..+22.76
#: range instead of probe one's 0..+12.
FRACTIONAL_SCALE = (-10.24, -5.24, 0.76, 4.76, 9.76, 13.76, 18.76, 22.76)


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, semitones: float, capped: bool) -> list[dict]:
    """One note as `compile.py` writes it, with or without the Note Off cap.

    The capped form is the exact event set the failing export uses for 307 of
    its notes: pitch, gain, start, then a release fade finishing at the cap and
    a stop on the wildcard channel at the same instant.
    """
    events = [
        fade_pitch(time_ms, semitones, 0),
        fade(time_ms, 0, 0.0),
        start(SAMPLE, time_ms, START_CHANNEL),
    ]
    if capped:
        end = time_ms + CAP_MS
        release_start = max(time_ms, end - RELEASE_MS)
        events.append(fade(release_start, -60.0, (end - release_start) / 1000.0))
        events.append(stop(end))
    return events


def _lanes(doc, groups, label: str, base_ms: int, pitches, capped: bool, repeats: int = 1):
    """Deal `pitches` round-robin across EMITTERS, exactly as the compiler does."""
    lanes: list[list[dict]] = [[] for _ in range(EMITTERS)]
    index = 0
    for _ in range(repeats):
        for semitones in pitches:
            lanes[index % EMITTERS].extend(_note(base_ms + index * GAP_MS, semitones, capped))
            index += 1
    for lane_index, events in enumerate(lanes):
        entity = doc.add_timeline()
        entity["displayName"] = f"{label} lane {lane_index}"
        groups.append((_timeline_ref(entity), events))
    return base_ms + index * GAP_MS


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    # ---- Group 1, 0-2 s: probe one's clean scale, now WITH the Note Off cap.
    # Only the cap changed.
    #   EXPECT: the same clean rising scale, each note just shorter.
    #   IF WRONG: the release fade or the wildcard stop is reaching a note it
    #   does not own. That is a compiler bug in the capped-note branch, and it
    #   would explain the fault directly -- 307 of the song's 385 notes take it.
    _lanes(doc, groups, "G1 capped whole scale", 0, WHOLE_SCALE, capped=True)

    # ---- Group 2, 4-6 s: probe one's UNcapped scale, now at the song's real
    # fractional pitches over its real range. Only the pitch values changed.
    #   EXPECT: a rising figure, wider than group 1 and slightly out of tune to
    #   the ear, but ordered low-to-high and steady.
    #   IF WRONG: fractional or wide-range modifiers do not survive, and the
    #   fault is in what we ask for rather than how we schedule it.
    _lanes(doc, groups, "G2 uncapped fractional", 4000, FRACTIONAL_SCALE, capped=False)

    # ---- Group 3, 8-10 s: both together, still only eight notes.
    #   EXPECT: whatever groups 1 and 2 each did, combined.
    #   IF WRONG only here: the two interact, which neither alone would show.
    _lanes(doc, groups, "G3 capped fractional", 8000, FRACTIONAL_SCALE, capped=True)

    # ---- Group 4, 12-35 s: the failing song in miniature -- 168 notes at the
    # song's spacing, capped, fractional, four emitters, run long enough for
    # emitter-slot recycling to bite.
    #   EXPECT: the same eight-note figure repeating 21 times, unchanged from
    #   the first repeat to the last.
    #   IF IT DEGRADES PART-WAY: the fault is sustained load, not any single
    #   note's events -- the engine is recycling slots out from under notes the
    #   allocator still believes it owns, which is exactly the failure
    #   `voices.py` warns about and which only appears at length.
    _lanes(doc, groups, "G4 sustained load", 12000, FRACTIONAL_SCALE, capped=True, repeats=21)

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-load-probe")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize(doc.data))
    print(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "examples"
        / "pitch_load_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
