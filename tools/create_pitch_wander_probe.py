"""Find why an exported song's pitches wander in game when the browser agrees.

The occasion: `DoomHangar.mid` on `Play_sfx_stranglerstrings_02`. Every offline
check says the export is correct -- the 385 written `fadePitch` values match the
browser preview exactly, each bound to the right note at the right millisecond,
one recording behind the event, the reported sample length matching the decoded
audio, and no emitter reused while its previous note still rings. It still
plays with the pitches scattered in game.

So the remaining difference is engine behavior under that song's real shape, and
these groups isolate it one variable at a time. Each states what it expects and
what a failure would prove; nothing here is a guess to be argued about
afterwards.

Deliberately the SAME sound and the SAME note spacing as the failing song
(275 ms one-shot, notes every 136 ms, four emitters), because the offline
evidence has already cleared everything that does not depend on those.
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

#: The song's own sound: one recording, 275 ms, non-looping.
SAMPLE = "Play_sfx_stranglerstrings_02"

#: A second, unmistakably tonal sound. Whatever the groups below show, running
#: the same pattern on this one says whether the answer is about the pattern or
#: about that particular recording.
CONTROL_SAMPLE = "play_pianoc4"

#: The failing song's median gap between note starts. Shorter than the 275 ms
#: sample, which is the whole point: consecutive notes overlap.
GAP_MS = 136

#: What the song really uses. Four emitters, round-robin, is the condition the
#: offline check said was safe.
EMITTERS = 4


def _timeline_ref(entity: dict) -> str:
    return entity["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"][
        "item[0]"
    ]["entity"]


def _note(time_ms: int, semitones: float, sample: str = SAMPLE) -> list[dict]:
    """One note exactly as `compile.py` writes it on an isolated emitter.

    Same event kinds, same order, same channels. A probe that wrote them any
    other way would answer a question nobody asked.
    """
    return [
        fade_pitch(time_ms, semitones, 0),
        fade(time_ms, 0, 0.0),
        start(sample, time_ms, START_CHANNEL),
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

    # ---- Group 1, 0-5 s: does pitch work at all on this sound, in isolation?
    # Three notes 1.5 s apart on one emitter, so no sample can still be ringing
    # when the next begins.
    #   EXPECT: three clearly different pitches -- low, middle, high.
    #   IF NOT: pitch application itself is broken for this sound, and nothing
    #   about density or emitters matters. Everything below becomes moot.
    emitter(
        "G1 isolated: low, mid, high (1.5 s apart)",
        [*_note(0, -12), *_note(1500, 0), *_note(3000, 12), stop(5000)],
    )

    # ---- Group 2, 6-8 s: the same three pitches at the SONG'S spacing, on ONE
    # emitter, so each note starts while the previous 275 ms sample still rings.
    #   EXPECT (if clean): still three distinct rising pitches, just fast.
    #   IF WRONG: a modifier is reaching the previous note's still-audible tail
    #   -- the notes smear or land on one pitch. That confirms same-emitter tail
    #   bleed at real density, which the offline reservation check said could
    #   not happen, and means the allocator's ring-length model is wrong.
    emitter(
        "G2 one emitter, 136 ms apart: low, mid, high",
        [
            *_note(6000, -12),
            *_note(6000 + GAP_MS, 0),
            *_note(6000 + 2 * GAP_MS, 12),
            stop(8000),
        ],
    )

    # ---- Group 3, 9-11 s: three DIFFERENT emitters, all struck at once, each
    # asking for its own pitch. This is the question doom-re's probe never
    # tested: its groups all lived on a single emitter.
    #   EXPECT (if per-emitter): a three-note chord, low + middle + high.
    #   IF WRONG: one pitch for all three, or wandering -- proving a modifier
    #   reaches sounds on OTHER emitters, which is exactly what would scramble a
    #   four-emitter song while the browser (isolated buffers) sounds perfect.
    for index, semitones in enumerate((-12, 0, 12)):
        emitter(f"G3 emitter {index}: simultaneous {semitones:+d}", [*_note(9000, semitones), stop(11000)])

    # ---- Group 4, 12-15 s: the failing song in miniature. A rising scale, one
    # note every 136 ms, dealt round-robin across four emitters exactly as the
    # compiler deals them.
    #   EXPECT (if clean): a recognisable rising scale.
    #   IF WRONG: this reproduces the reported fault in eight notes instead of
    #   385, and every later question can be asked against this map instead of
    #   against a whole song.
    scale = (0, 2, 4, 5, 7, 9, 11, 12)
    lanes: list[list[dict]] = [[] for _ in range(EMITTERS)]
    for index, semitones in enumerate(scale):
        lanes[index % EMITTERS].extend(_note(12000 + index * GAP_MS, semitones))
    for index, events in enumerate(lanes):
        emitter(f"G4 scale lane {index}", [*events, stop(15000)])

    # ---- Group 5, 16-19 s: the identical scale on a different sound.
    #   EXPECT: whatever group 4 did, this does too.
    #   IF DIFFERENT: the fault belongs to `Play_sfx_stranglerstrings_02`
    #   specifically rather than to the scheduling pattern, and the search moves
    #   back to that recording.
    control_lanes: list[list[dict]] = [[] for _ in range(EMITTERS)]
    for index, semitones in enumerate(scale):
        control_lanes[index % EMITTERS].extend(
            _note(16000 + index * GAP_MS, semitones, CONTROL_SAMPLE)
        )
    for index, events in enumerate(control_lanes):
        emitter(f"G5 control-sound scale lane {index}", [*events, stop(19000)])

    entity_events = {
        f"item[{index}]": {"entity": entity_ref, "events": events_block(events)}
        for index, (entity_ref, events) in enumerate(groups)
    }
    entity_events["num"] = len(groups)
    master["entityDef"]["state"]["edit"]["componentTimeLine"]["entityEvents"] = entity_events
    add_button(doc, master_ref, "pitch-wander-probe")

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
        / "pitch_wander_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
