"""Third pass: find WHAT runs out when pitched notes degrade under load.

Probe two reproduced the reported fault. Its groups 1-3 (eight notes each, with
the Note Off cap, with fractional and wide-range pitch) all played correctly;
group 4 -- the identical figure repeated 21 times -- started clean and then
scattered. So neither the events, the cap, nor the pitch values are wrong. Some
resource is exhausting while the song runs.

`voices.py` names the suspected mechanism: the engine recycles sound emitter
slots under load, and a one-shot whose slot is recycled can no longer be
stopped, so it rings its whole sample while later `fadePitch` events -- which
address whatever is audible -- reach it.

Each group below runs the SAME number of notes and changes exactly one
resource, so whichever group survives names the thing that ran out. Every group
uses the failing song's own fractional pitches, and the figure is deliberately
repetitive: the moment it stops repeating cleanly is the answer.
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

CAP_MS = 230
RELEASE_MS = 100

#: Same count in every group, so the only difference is the resource under test.
NOTES = 48

#: The failing export's own values.
FRACTIONAL_SCALE = (-10.24, -5.24, 0.76, 4.76, 9.76, 13.76, 18.76, 22.76)


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, semitones: float, capped: bool) -> list[dict]:
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


def _run(doc, groups, label, base_ms, *, emitters, gap_ms, capped):
    """`NOTES` notes dealt round-robin, one resource varied per call."""
    lanes: list[list[dict]] = [[] for _ in range(emitters)]
    for index in range(NOTES):
        semitones = FRACTIONAL_SCALE[index % len(FRACTIONAL_SCALE)]
        lanes[index % emitters].extend(_note(base_ms + index * gap_ms, semitones, capped))
    for lane_index, events in enumerate(lanes):
        entity = doc.add_timeline()
        entity["displayName"] = f"{label} lane {lane_index}"
        groups.append((_timeline_ref(entity), events))
    return base_ms + NOTES * gap_ms


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    # ---- Group 1: the baseline, matching the song and probe two's group 4.
    #   EXPECT: degrades, the same way probe two did. If it does NOT, the fault
    #   needs more than 48 notes and every comparison below shifts with it.
    end = _run(doc, groups, "G1 baseline 4 emitters", 0, emitters=4, gap_ms=136, capped=True)

    # ---- Group 2: same notes, same rate, EIGHT emitters instead of four. Each
    # emitter now handles half as many notes and gets twice as long between
    # them.
    #   IF CLEAN: what ran out is per-emitter -- slots being reused before the
    #   engine has really released the previous note. The fix is then ours:
    #   reserve emitters for longer, or spread notes across more of them.
    #   IF IT STILL DEGRADES: the limit is global, not per-emitter, and adding
    #   emitters cannot help.
    end = _run(doc, groups, "G2 eight emitters", end + 2000, emitters=8, gap_ms=136, capped=True)

    # ---- Group 3: same notes, same four emitters, but HALF the rate.
    #   IF CLEAN: the limit is about notes per second, not notes total -- the
    #   engine needs recovery time between starts.
    #   IF IT STILL DEGRADES: elapsed time does not matter and the count does.
    end = _run(doc, groups, "G3 half rate", end + 2000, emitters=4, gap_ms=272, capped=True)

    # ---- Group 4: the baseline again with NO release fade and NO stop, so each
    # one-shot simply rings out.
    #   IF CLEAN: our stops are the problem -- a wildcard stop is reaching
    #   something it should not, and the capped-note branch needs rewriting.
    #   IF IT STILL DEGRADES: stops are innocent, and the exhaustion is in
    #   starting sounds this fast regardless of how they end.
    _run(doc, groups, "G4 no stops", end + 2000, emitters=4, gap_ms=136, capped=False)

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-exhaustion-probe")

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
        / "pitch_exhaustion_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
