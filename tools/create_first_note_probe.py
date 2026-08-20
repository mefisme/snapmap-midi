"""Pin down the first-note-plays-raw fault, in the simplest form that can show it.

Probe three came back with one clear pattern: the opening notes of every group
played at the sample's natural pitch, and the count that came out raw tracked
the number of emitters -- four emitters, about four raw notes; eight emitters,
about six to eight. Probe two's fractional group showed the same thing, where
the first note on each of its four lanes was raw and the second was correct.

That points at the first `fadePitch` on a given emitter not reaching its note.
Everything here is built to make that unmissable rather than plausible:

  - **+24 semitones**, two full octaves, so "pitched" and "raw" cannot be
    confused with each other or with a fractional detune;
  - **500 ms spacing**, so nothing overlaps and each note can be counted;
  - **every note asking for the SAME pitch**, so any note that differs from its
    neighbours is the fault itself rather than a melody to interpret.

Group 1 alone settles it. Groups 2-4 say whether it is per emitter, whether the
pitch is arriving one note late, and whether speed matters.
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

#: A plain piano note, NOT the song's strings sample.
#:
#: The listener has to answer "is this note the same pitch as the last one",
#: and that is far harder on a noisy, atonal recording than on a piano. Probe
#: one already showed the two behave alike (its group 5 ran the same scale on
#: this sound), so nothing about the fault is being hidden by the swap -- only
#: the difficulty of hearing it.
SAMPLE = "play_pianoc4"

#: Two octaves up. The engine accepts +24 (doom-re probe, group 6), and nothing
#: about a raw sample can be mistaken for it.
HIGH = 24.0

#: Slow enough to count the notes out loud.
SLOW_MS = 700
FAST_MS = 136


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, semitones: float) -> list[dict]:
    """Exactly the shape `compile.py` writes: pitch, gain, then start."""
    return [
        fade_pitch(time_ms, semitones, 0),
        fade(time_ms, 0, 0.0),
        start(SAMPLE, time_ms, START_CHANNEL),
    ]


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    def emitter(name: str, events: list[dict]) -> None:
        entity = doc.add_timeline()
        entity["displayName"] = name
        groups.append((_timeline_ref(entity), events))

    # ---- Group 1, 0-5 s: ONE emitter, eight notes, every one asking +24.
    #   EXPECT if healthy: eight identical high notes.
    #   EXPECT if the fault is real: the FIRST note low (raw) and the remaining
    #   seven high. One low note at the front is the entire finding.
    emitter(
        "G1 one emitter, 8x +24",
        [event for i in range(8) for event in _note(i * SLOW_MS, HIGH)] + [stop(5600)],
    )

    # ---- Group 2, 9-12 s: FOUR emitters, one note each, all asking +24.
    #   IF ALL FOUR ARE RAW: the loss is per emitter -- every emitter drops its
    #   own first modifier, so a four-emitter song loses four notes and an
    #   eight-emitter song loses eight. That matches probe three exactly.
    #   IF ALL FOUR ARE HIGH: the loss is not per emitter and group 1's first
    #   note failed for a different reason.
    for index in range(4):
        emitter(
            f"G2 emitter {index}, single +24",
            [*_note(9000 + index * SLOW_MS, HIGH), stop(12000)],
        )

    # ---- Group 3, 15-21 s: ONE emitter alternating raw and +24.
    #   EXPECT if healthy: low, high, low, high, low, high, low, high.
    #   EXPECT if the pitch arrives ONE NOTE LATE: low, low, high, low, high,
    #   low, high, low -- the pattern shifted by one. This is what separates
    #   "the first modifier is dropped" from "every modifier lands on the next
    #   note instead of its own", which sound the same in group 1 but need
    #   completely different fixes.
    alternating = []
    for index in range(8):
        alternating.extend(_note(15000 + index * SLOW_MS, 0.0 if index % 2 == 0 else HIGH))
    emitter("G3 one emitter, alternating raw/+24", [*alternating, stop(20600)])

    # ---- Group 4, 24-26 s: group 1 again at the song's real spacing.
    #   IF the same single raw note appears: speed is irrelevant and the fault
    #   is purely about the first modifier on an emitter.
    #   IF MORE notes come out raw: speed compounds it, and the fix has to
    #   consider rate as well.
    emitter(
        "G4 one emitter, 8x +24 at song speed",
        [event for i in range(8) for event in _note(24000 + i * FAST_MS, HIGH)] + [stop(26000)],
    )

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "first-note-probe")

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
        / "first_note_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
