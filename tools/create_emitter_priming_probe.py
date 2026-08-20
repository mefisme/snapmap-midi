"""Does priming an emitter make its FIRST note accept a pitch?

Established by the timing probe, in game:

  - a `fadePitch` written even 2 ms before its note is lost entirely -- the
    whole run played raw. A modifier must share the exact timestamp of the
    `startSoundShader` it belongs to. (Consistent with the doom-re finding that
    a modifier addresses what is audible at the instant it fires plus a sound
    started at that same instant: fired into silence, it has nothing to attach
    to and nothing later inherits it.)
  - with the modifier at the correct same timestamp, the first four or five
    notes still played at the sample's natural pitch and everything after was
    correct. The run used FOUR emitters. Two earlier probes reported the same
    thing ("the first 4-6 notes were just the raw pitch"), which was set aside
    at the time.

So an emitter appears to ignore the first modifier it is ever given, and to
honour every one after that. Four emitters, four wrong notes -- and in a real
song that is four wrong notes at the start plus another wherever the engine
recycles a slot, which is exactly the reported "random notes play raw".

If that reading is right, the cure is to spend each emitter's first modifier on
something inaudible before the music starts. Group B does that: one silent
primer per emitter, a full second ahead of the first real note, played at
-60 dB and stopped immediately.

Every note asks for +24, so a raw note is both lower and four times longer --
this probe hunts a MISSING pitch, not a wrong one.
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

GAP_MS = 136
EMITTERS = 4
NOTES = 32
CAP_MS = 250
RELEASE_MS = 100

#: Silent, and far enough ahead that it cannot be confused with the music.
PRIMER_LEAD_MS = 1000
PRIMER_DB = -60.0


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


def _primer(time_ms: int) -> list[dict]:
    """Spend this emitter's first modifier where nobody can hear it.

    Same event shape as a real note so it exercises the identical path, but
    started at -60 dB and stopped 20 ms later.
    """
    return [
        fade_pitch(time_ms, PITCH, 0),
        fade(time_ms, PRIMER_DB, 0.0),
        start(SAMPLE, time_ms, START_CHANNEL),
        stop(time_ms + 20),
    ]


def _run(doc, groups, label: str, base_ms: int, *, primed: bool) -> int:
    lanes: list[list[dict]] = [[] for _ in range(EMITTERS)]
    if primed:
        for lane in range(EMITTERS):
            lanes[lane].extend(_primer(base_ms - PRIMER_LEAD_MS + lane * 5))
    for index in range(NOTES):
        lanes[index % EMITTERS].extend(_note(base_ms + index * GAP_MS))
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

    # ---- Group A, 1.5-6 s: unprimed. The control.
    #   EXPECT: the first four notes raw (low and long), the rest high.
    end = _run(doc, groups, "A unprimed", 1500, primed=False)

    # ---- Group B, 9-14 s: one silent primer per emitter, a second early.
    #   IF ALL 32 ARE HIGH: confirmed, and the fix is four inaudible events per
    #   emitter in `compile.py`.
    #   IF THE FIRST FOUR ARE STILL RAW: priming does not transfer -- the
    #   emitter is not "warmed" by a previous sound, and something about the
    #   very first modifier in the map is being dropped instead.
    _run(doc, groups, "B primed", end + 3000, primed=True)

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "emitter-priming-probe")

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
        / "emitter_priming_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
