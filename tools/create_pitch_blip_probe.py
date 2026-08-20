"""Can the modifier sit on the start's own timestamp, but AFTER it?

The race fix works -- every note now receives its pitch -- but it costs one
millisecond of natural pitch at the head of each note, and on a percussive
sample that millisecond is the attack, so it is audible as a blip.

`eventTime` cannot go finer: a game-authored map carries 8,474 of them and
every one is an integer, so 1 ms is the format's resolution. The remaining
freedom is ORDER WITHIN a timestamp, and only two of the three possibilities
have been tested:

  - modifier BEFORE the start, same timestamp -- racy. It depends on the
    engine's special case for a not-yet-started sound, and when the start won
    instead the note played raw (measured at roughly 2 notes in 11).
  - modifier 1 ms AFTER the start -- reliable, because doom-re established a
    modifier reaches a sound that is already playing. This is what ships now,
    and what produces the blip.
  - modifier on the SAME timestamp but written after the start in the events
    array -- untested. If the engine processes equal-time events in array
    order, the sound is already playing when the modifier runs, so it should be
    as reliable as the 1 ms version with no gap at all.

Group A is what ships today. Group B is the third option. Both play 24 notes at
+24 through the song's own sound, so the two questions are simply: does every
note get its pitch, and can you hear a blip at the front of it.
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
NOTES = 24
GAP_MS = 300


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, *, same_tick: bool) -> list[dict]:
    """One note with the modifier either 1 ms after the start, or on its tick.

    Both forms put the modifier after the start in the array; they differ only
    in whether it also carries a later timestamp. The sort in `_run` is stable,
    so the same-tick form keeps the start ahead of the modifier.
    """
    end = time_ms + CAP_MS
    release_start = max(time_ms, end - RELEASE_MS)
    pitch_at = time_ms if same_tick else time_ms + 1
    return [
        fade(time_ms, 0, 0.0),
        start(SAMPLE, time_ms, START_CHANNEL),
        fade_pitch(pitch_at, PITCH, 0),
        fade(release_start, -60.0, (end - release_start) / 1000.0),
        stop(end),
    ]


def _run(doc, groups, label, base_ms, *, same_tick):
    events: list[dict] = []
    for index in range(NOTES):
        events.extend(_note(base_ms + index * GAP_MS, same_tick=same_tick))
    entity = doc.add_timeline()
    entity["displayName"] = label
    groups.append((_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"])))
    return base_ms + NOTES * GAP_MS


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    # ---- Group A: what ships now -- modifier one millisecond later.
    #   EXPECT: all 24 notes pitched, each with a faint blip at its front.
    end = _run(doc, groups, "A pitch 1 ms after start", 0, same_tick=False)

    # ---- Group B: modifier on the start's own tick, after it in the array.
    #   IF ALL PITCHED AND NO BLIP: array order is honoured within a timestamp,
    #   and this is strictly better than what ships -- same reliability, no gap.
    #   IF NOTES GO RAW AGAIN: equal-time order is not dependable in this
    #   direction either, the 1 ms version is the only safe one, and the blip
    #   has to be solved by starting the note silent instead.
    _run(doc, groups, "B pitch on the start tick", end + 3000, same_tick=True)

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-blip-probe")

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
        / "pitch_blip_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
