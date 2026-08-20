"""Find the note count at which pitched playback starts to fall apart.

Every mechanism guessed at so far has been eliminated by probe:

  - pitch works on an isolated note, and on simultaneous notes across separate
    emitters (wander probe, first-note probe group 2);
  - the Note Off cap, its release fade and its stop are all fine (load probe
    group 1);
  - fractional and wide-range values are fine (load probe groups 2-3);
  - a `stopSound` does NOT reach other emitters (stop-scope probe: the
    un-stopped high note rang on after the other emitter stopped itself);
  - notes bending each other needs them to overlap on one emitter, which the
    failing song never does -- 275 ms sample, capped at 230 ms, 544 ms between
    that emitter's notes. The probe that showed bleeding used a 4,972 ms piano
    sample restarted every 136 ms, which was the probe's own doing.

What is left is quantity. The same figure is clean at 8 notes and scattered at
168. This finds where in between it turns, which is the one fact needed before
any fix can be chosen -- a per-emitter limit, a global voice pool and a rate
limit all predict different numbers, and none of them can be told apart by
argument.

Four blocks, doubling each time, identical in every other respect. The notes
alternate an octave down and an octave up, so a clean block is a perfectly
even low-high-low-high warble and a faulty one audibly loses that regularity.
There is no melody to follow and nothing to count precisely: the answer wanted
is only "which block was the first bad one".
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

#: The failing song's own sound: 275 ms, one recording, non-looping.
SAMPLE = "Play_sfx_stranglerstrings_02"

#: ALTERNATING low and high, one octave either side of natural.
#:
#: The first version of this probe put every note at the same pitch, and all
#: four blocks came back sounding identical -- necessarily, because a modifier
#: leaking onto its neighbour would have set that neighbour to the value it
#: already had. A same-pitch run cannot detect pitch bleed at all.
#:
#: Alternating makes any leak audible instantly: the run should be a perfectly
#: regular low-high-low-high warble, and a modifier reaching the wrong note
#: breaks that regularity. It also matches what the failing groups had and the
#: clean ones did not -- every group that misbehaved used VARYING pitches.
LOW_PITCH = -12.0
HIGH_PITCH = 12.0

#: The song's real numbers.
GAP_MS = 136
EMITTERS = 4
CAP_MS = 230
RELEASE_MS = 100

#: 64 was already proven clean and 168 was already seen to degrade, so the
#: threshold sits between them. 385 is the failing song's own note count.
BLOCKS = (8, 16, 32, 64)

#: Long enough that a block's last note has fully died before the next begins.
SILENCE_MS = 2000


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, semitones: float) -> list[dict]:
    """Exactly what `compile.py` writes for a Note Off capped one-shot."""
    end = time_ms + CAP_MS
    release_start = max(time_ms, end - RELEASE_MS)
    return [
        fade_pitch(time_ms, semitones, 0),
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

    cursor = 0
    for block_index, count in enumerate(BLOCKS, start=1):
        lanes: list[list[dict]] = [[] for _ in range(EMITTERS)]
        for index in range(count):
            semitones = LOW_PITCH if index % 2 == 0 else HIGH_PITCH
            lanes[index % EMITTERS].extend(_note(cursor + index * GAP_MS, semitones))
        for lane_index, events in enumerate(lanes):
            entity = doc.add_timeline()
            entity["displayName"] = f"B{block_index} {count} notes lane {lane_index}"
            groups.append(
                (_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"]))
            )
        cursor += count * GAP_MS + SILENCE_MS

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "note-count-probe")

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
        / "note_count_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
