"""Does a same-millisecond `fadePitch` reliably reach its own note?

The reported symptom is exact and unusual: under a real song, RANDOM notes play
the sample at its natural pitch. Not a wrong pitch -- no pitch. Per
`doom-re/docs/truth/engine/snapmap-timeline-sound-modifiers.md` group 2 the
modifier is per-event and is NOT retained emitter state, so "plays natural"
means precisely one thing: that note's `fadePitch` did not take effect before
its `startSoundShader`.

`compile.py` writes both at the SAME `eventTime` and depends on array order to
put the modifier first. Group 1 of that same probe established the ordering
works -- with two events, on one emitter, once. Nothing has ever tested whether
it still holds when a song is firing five events per note across four emitters
several times a second.

Group A is what the compiler writes today. Group B changes one thing: the
modifier goes two milliseconds earlier, so it can no longer tie with its own
start and the engine has no ordering decision left to make.

Every note asks for +24. That is deliberate: this probe is not looking for a
wrong pitch, it is looking for a MISSING one, and against a run of identical
high notes a raw note is unmistakable -- lower, and much longer, because the
sample is not being sped up.
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

#: The song's own sound.
SAMPLE = "Play_sfx_stranglerstrings_02"

#: Two octaves up: a raw note is both lower AND four times longer, so it stands
#: out twice over.
PITCH = 24.0

#: The song's real shape.
GAP_MS = 136
EMITTERS = 4
NOTES = 96
CAP_MS = 250
RELEASE_MS = 100

#: Far enough ahead that the modifier cannot tie with its own start, close
#: enough that it cannot reach the previous note on that emitter (which is
#: 544 ms earlier and already stopped).
LEAD_MS = 2


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, *, lead: int) -> list[dict]:
    end = time_ms + CAP_MS
    release_start = max(time_ms, end - RELEASE_MS)
    return [
        fade_pitch(time_ms - lead, PITCH, 0),
        fade(time_ms, 0, 0.0),
        start(SAMPLE, time_ms, START_CHANNEL),
        fade(release_start, -60.0, (end - release_start) / 1000.0),
        stop(end),
    ]


def _run(doc, groups, label: str, base_ms: int, *, lead: int) -> int:
    lanes: list[list[dict]] = [[] for _ in range(EMITTERS)]
    for index in range(NOTES):
        lanes[index % EMITTERS].extend(_note(base_ms + index * GAP_MS, lead=lead))
    for lane_index, events in enumerate(lanes):
        entity = doc.add_timeline()
        entity["displayName"] = f"{label} lane {lane_index}"
        groups.append((_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"])))
    return base_ms + NOTES * GAP_MS


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    # ---- Group A: today's behaviour -- modifier and start share a timestamp.
    #   EXPECT if the ordering is safe: 96 identical short high notes.
    #   EXPECT if it is not: occasional low, long notes scattered through it,
    #   which is the reported fault reproduced.
    end = _run(doc, groups, "A same millisecond", 0, lead=0)

    # ---- Group B: the modifier two milliseconds ahead of its own start.
    #   IF CLEAN while A is not: the tie is the fault and the fix is one number
    #   in `compile.py`.
    #   IF BOTH ARE BAD: ordering is innocent and the modifier is being dropped
    #   for some other reason.
    #   IF BOTH ARE CLEAN: 96 notes is not enough to provoke it and the next
    #   step is length, not ordering.
    _run(doc, groups, "B two ms early", end + 3000, lead=LEAD_MS)

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-timing-probe")

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
        / "pitch_timing_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
