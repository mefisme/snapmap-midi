"""Write the modifier one millisecond AFTER its note instead of tied to it.

The startup-delay reading is dead: in that probe the notes at t=0 and t=300
were correct while the 3rd and 6th were raw. There is no positional pattern.
The failure is INTERMITTENT -- about two notes in eleven -- which is what the
original report said all along: "random notes it will play the sample at its
raw pitch".

Intermittent, on a single emitter, with nothing overlapping, points at a race.
`compile.py` writes `fadePitch` and `startSoundShader` at the SAME eventTime and
relies on their order in the saved array to put the modifier first. When the
start wins instead, that note has no modifier of its own and -- per doom-re
group 2, modifiers being per-event and never inherited -- plays at natural
pitch. Exactly the symptom, exactly its randomness.

Moving the modifier EARLIER is already disproven: at two milliseconds ahead the
whole run played raw, because a modifier fired into silence has no sound to
attach to and nothing later inherits it.

That leaves the other direction, which doom-re group 3 established and which
this codebase already trusts for glide ("One millisecond keeps the ramp on the
newly started sound"): a modifier reaches a sound that is ALREADY PLAYING. One
millisecond after the start there is no race left to lose -- the sound is
definitely there.

The cost is one millisecond of natural pitch at the very front of each note,
far below anything audible, against a note that currently has a real chance of
playing wrong for its entire length.

24 notes per group so a rate near two-in-eleven cannot hide, all asking +24,
one emitter, nothing overlapping.
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


def _note(time_ms: int, *, pitch_after: bool) -> list[dict]:
    """One note, with the modifier either tied to the start or 1 ms after it.

    The tied form must reproduce `compile.py` exactly, which means the
    modifier is appended FIRST so the stable sort leaves it ahead of the
    same-time start. Appending it last would make group A a different
    experiment rather than today's behaviour.
    """
    end = time_ms + CAP_MS
    release_start = max(time_ms, end - RELEASE_MS)
    if pitch_after:
        head = [
            fade(time_ms, 0, 0.0),
            start(SAMPLE, time_ms, START_CHANNEL),
            fade_pitch(time_ms + 1, PITCH, 0),
        ]
    else:
        head = [
            fade_pitch(time_ms, PITCH, 0),
            fade(time_ms, 0, 0.0),
            start(SAMPLE, time_ms, START_CHANNEL),
        ]
    return head + [
        fade(release_start, -60.0, (end - release_start) / 1000.0),
        stop(end),
    ]


def _run(doc, groups, label, base_ms, *, pitch_after):
    events: list[dict] = []
    for index in range(NOTES):
        events.extend(_note(base_ms + index * GAP_MS, pitch_after=pitch_after))
    entity = doc.add_timeline()
    entity["displayName"] = label
    # Stable sort keeps equal-time events in the order appended, which is what
    # makes group A a genuine reproduction of today's behaviour.
    groups.append((_timeline_ref(entity), sorted(events, key=lambda e: e["eventTime"])))
    return base_ms + NOTES * GAP_MS


def build(path: Path) -> None:
    doc = SnapMapDocument(blank_map(), palette_refs=PRODUCT_PALETTE_REFS)
    master = ensure_timeline(doc)
    master_ref = _timeline_ref(master)
    groups: list[tuple[str, list[dict]]] = [(master_ref, [])]

    # ---- Group A: today's behaviour, modifier tied to the start.
    #   EXPECT: a handful of raw notes scattered through, no pattern.
    end = _run(doc, groups, "A modifier at the start", 0, pitch_after=False)

    # ---- Group B: modifier one millisecond after the start.
    #   IF CLEAN: the race is the fault and the fix is a single "+ 1" in
    #   `compile.py`, on a line the glide path already proves is safe.
    #   IF STILL RANDOM: the modifier is being dropped for a reason unrelated
    #   to ordering, and no scheduling change will help.
    _run(doc, groups, "B modifier 1 ms after the start", end + 3000, pitch_after=True)

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-race-probe")

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
        / "pitch_race_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
