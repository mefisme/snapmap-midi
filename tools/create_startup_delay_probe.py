"""How long after a Timeline starts before a `fadePitch` reliably lands?

Two runs of the same events differed only in when they began, and only one was
faulty:

  - timing probe, notes from t=0: the first four or five played at the sample's
    natural pitch, the rest were correct;
  - priming probe, identical notes from t=1500: every note correct.

Earlier probes agree once read this way. Every group that reported "the first
4-6 notes were raw" began at or near the trigger; every group that came back
clean began seconds into the map. What looked like "the first note on each
emitter" was really "the first notes after the Timeline starts" -- the two are
indistinguishable when a round-robin deals its opening notes across all
emitters at once, which is what those probes did.

The failing song's first note is at t=0.

This measures the boundary directly. Eleven notes on ONE emitter -- so emitter
count cannot confound it -- at widening offsets from the trigger, every one
asking +24. Counting how many are low at the front gives the lead-in the
compiler must leave, instead of picking a padding number by guesswork.
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
PITCH = 24.0
CAP_MS = 250
RELEASE_MS = 100

#: Offsets from the Timeline trigger. Tight at the front where the boundary is
#: expected, widening after, and spaced so no note can overlap the next.
OFFSETS = (0, 300, 600, 900, 1200, 1600, 2000, 2500, 3000, 3600, 4200)


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int) -> list[dict]:
    end = time_ms + CAP_MS
    release_start = max(time_ms, end - RELEASE_MS)
    return [
        fade_pitch(time_ms, PITCH, 0),
        fade(time_ms, 0, 0.0),
        start(SAMPLE, time_ms, START_CHANNEL),
        fade(release_start, -60.0, (end - release_start) / 1000.0),
        stop(end),
    ]


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    # One emitter for the whole run. Whatever the answer is, it cannot be about
    # how many emitters are involved.
    #   EXPECT if the boundary is real: the opening notes low and long, turning
    #   high at some point in the sequence. The offset where that happens is the
    #   lead-in the compiler has to leave before the first note of a song.
    #   EXPECT if it is not: eleven identical high notes, and the earlier
    #   difference came from something other than start time.
    events: list[dict] = []
    for offset in OFFSETS:
        events.extend(_note(offset))
    entity = doc.add_timeline()
    entity["displayName"] = "startup delay, one emitter, all +24"
    groups.append((_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"])))

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "startup-delay-probe")

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
        / "startup_delay_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
