"""Does a `stopSound` from one emitter silence notes on OTHER emitters?

Nothing has tested this, and it is the last mechanism standing.

What is already settled by probe:

  - a note alone on an emitter always gets its pitch (first-note probe, group 2);
  - notes piled onto ONE emitter bend each other -- but that only happened
    because `play_pianoc4` rings for 4,972 ms and the probe restarted it every
    136-700 ms. The failing song cannot do this: a 275 ms sample, capped at
    230 ms, with 544 ms between notes on any one emitter;
  - the failing figure is clean over 8 notes and scatters over 168 (load probe),
    so whatever is wrong ACCUMULATES rather than being wrong per note.

The song writes 307 `stopSound` events, all on `SND_CHANNEL_ANY`, spread across
four emitters firing one every 544 ms. If that wildcard stop is scoped to the
whole map rather than to the emitter that fired it, then every emitter is
constantly cutting the other three -- notes vanish, survivors are left ringing
where the bookkeeping says they are gone, and it gets worse the more emitters
and notes there are. That matches everything observed, including the eight
emitter group of the exhaustion probe sounding worse rather than better.

The test needs no counting and no melody. Two notes, two emitters, one stop.

`play_pianoc4` is used precisely because it rings for five seconds: it gives
the second note something long enough to still be audible when the first
emitter's stop fires.
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

SAMPLE = "play_pianoc4"

#: 4,972 ms at natural pitch, so an octave up still rings about 2.5 s -- long
#: enough to outlive the stop fired at 1.5 s below.
LOW = 0.0
HIGH = 12.0


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, semitones: float) -> list[dict]:
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
        groups.append((_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"])))

    # ---- Group 1, 0-3 s: the question itself.
    # Emitter A holds a LOW note from 0 and stops ITSELF at 1.5 s.
    # Emitter B holds a HIGH note from 0.5 s, rings until about 3.0 s, and is
    # never told to stop by anyone.
    #   IF STOPS ARE PER-EMITTER: at 1.5 s the low note disappears and the high
    #   note keeps ringing alone for another second and a half.
    #   IF STOPS ARE GLOBAL: everything goes silent together at 1.5 s, and that
    #   is the fault -- the song's four emitters fire 307 of these at each
    #   other, which is exactly an accumulating scatter that gets worse with
    #   more emitters and more notes.
    emitter("G1-A low, stops itself at 1.5 s", [*_note(0, LOW), stop(1500)])
    emitter("G1-B high, never stopped", _note(500, HIGH))

    # ---- Group 2, 6-11 s: the identical pair with NO stop anywhere.
    # This is the control for group 1: it proves the high note really is
    # audible past the 1.5 s mark on its own, so a silence in group 1 can only
    # have come from the stop.
    #   EXPECT: both ring out, the high note fading first, the low note
    #   continuing alone afterwards.
    emitter("G2-A low, no stop", _note(6000, LOW))
    emitter("G2-B high, no stop", _note(6500, HIGH))

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "stop-scope-probe")

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
        / "stop_scope_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
