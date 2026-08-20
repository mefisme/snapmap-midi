"""Five pitches, five emitters, one second apart: does the engine go DOWN?

`doom-re/docs/truth/engine/snapmap-timeline-sound-modifiers.md` confirmed
0, +12, +21 and +24 as four distinct rising pitches, and its Boundaries section
states plainly: "The negative pitch range (below 0) was not tested."

Everything observed in the failing song points at that gap:

  - its bottom note is written `+0.00` and sounds correct -- but "correct" and
    "unmodified" are the same thing at zero, so that proves nothing;
  - the notes that need an actual shift are the ones that come out wrong;
  - in the load probe's fractional group the first four notes -- -10.24, -5.24,
    +0.76 and +4.76 -- were reported as "the raw pitch not tuned to anything",
    while the later, larger, positive values "eventually rose".

If downward modifiers are ignored or clamped, a melody keeps only the notes
above its reference and flattens everything below onto natural pitch, which is
what "the bottom note is right and the others are wrong" sounds like.

The design removes every confound already ruled out: one note per emitter (a
configuration proven clean), a full second between notes, each stopped before
the next begins, and two-octave steps so the direction is unmistakable. Nothing
overlaps, nothing is reused, nothing has to be counted.
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

#: Piano: unambiguously tonal, so "higher or lower" needs no expertise.
SAMPLE = "play_pianoc4"

#: Bottom to top of the engine's stated range, in even steps.
STEPS = (-24.0, -12.0, 0.0, 12.0, 24.0)

SPACING_MS = 1200
#: Each note is silenced before the next starts, so nothing can bend anything.
HOLD_MS = 1000


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    def emitter(name: str, events: list[dict]) -> None:
        entity = doc.add_timeline()
        entity["displayName"] = name
        groups.append((_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"])))

    # ---- Group 1, 0-6 s: rising. -24, -12, 0, +12, +24.
    #   EXPECT if healthy: five clearly rising pitches, evenly spaced.
    #   EXPECT if downward modifiers are lost: the first three all sound the
    #   SAME (natural), then the last two rise. That is the fault, and it
    #   explains the song exactly -- everything below the reference collapses
    #   onto one pitch while everything above it plays correctly.
    for index, semitones in enumerate(STEPS):
        at = index * SPACING_MS
        emitter(
            f"G1 rising {semitones:+g}",
            [
                fade_pitch(at, semitones, 0),
                fade(at, 0, 0.0),
                start(SAMPLE, at, START_CHANNEL),
                stop(at + HOLD_MS),
            ],
        )

    # ---- Group 2, 8-14 s: the same five values in DESCENDING order.
    # Order is the only change. A listener who has just heard group 1 rise can
    # be primed to hear a rise again; if group 2 does not fall, the ear is not
    # what decided group 1.
    #   EXPECT if healthy: five clearly falling pitches.
    #   EXPECT if downward modifiers are lost: it falls for two notes and then
    #   sticks on natural pitch for the last three.
    for index, semitones in enumerate(reversed(STEPS)):
        at = 8000 + index * SPACING_MS
        emitter(
            f"G2 falling {semitones:+g}",
            [
                fade_pitch(at, semitones, 0),
                fade(at, 0, 0.0),
                start(SAMPLE, at, START_CHANNEL),
                stop(at + HOLD_MS),
            ],
        )

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-direction-probe")

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
        / "pitch_direction_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
