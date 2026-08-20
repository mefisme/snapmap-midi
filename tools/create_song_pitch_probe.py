"""Take the failing song's own export and make its pitches judgeable by ear.

Every synthetic probe so far has come back clean at the song's spacing, its
emitter count, its cap, its sound, whole and fractional pitch values, and up to
64 notes -- while the song itself scatters. Rather than keep guessing which
remaining property matters, this rewrites the REAL export and changes exactly
one thing: every `fadePitch` value becomes an alternating octave down / octave
up, in time order.

Everything else is the song's own: 385 notes, its irregular gaps down to 14 ms,
its four emitters and how the allocator dealt notes between them, which 74
notes go uncapped, its event counts per entity, its 53-second length.

That makes the result decisive either way:

  - **an even low-high warble throughout** -- the scheduling, density and length
    of this song are all fine, and the fault is in the pitch VALUES it asks for
    (24 distinct fractional values spanning -10.24..+22.76). The search moves to
    which values, and `create_pitch_fraction_probe.py` already exists for that.
  - **ragged, gliding, or notes at natural pitch** -- the song's own structure
    provokes the fault where every synthetic reproduction of it did not, and the
    difference between this map and the clean probes is a short, finite list to
    bisect.

Run `create_song_pitch_probe.py <settings.json>` against the sidecar; it
compiles the song exactly as Export would, then rewrites only the pitch values.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from snapmap_midi.ui.session import Session

#: One octave either side of natural. Big enough that a wrong note is obvious,
#: and symmetric so neither direction is favoured.
LOW_PITCH = -12.0
HIGH_PITCH = 12.0

_EVENT_KEY = "\neventHandle_t eventDef"


def _iter_event_blocks(node):
    """Yield every Timeline `events` block in the document."""
    if isinstance(node, dict):
        events = node.get("events")
        if isinstance(events, dict) and any(k.startswith("item[") for k in events):
            yield events
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _iter_event_blocks(value)
    elif isinstance(node, list):
        for value in node:
            if isinstance(value, (dict, list)):
                yield from _iter_event_blocks(value)


def _pitch_events(data):
    """Every fadePitch call in the map, tagged with the time it fires."""
    found = []
    for block in _iter_event_blocks(data):
        for key in block:
            if not key.startswith("item["):
                continue
            event = block[key]
            if not isinstance(event, dict):
                continue
            call = event.get("eventCall")
            if isinstance(call, dict) and call.get(_EVENT_KEY) == "fadePitch":
                found.append((event["eventTime"], call))
    return found


def build(settings_path: Path, output: Path) -> None:
    document = json.loads(settings_path.read_text(encoding="utf-8"))
    session = Session(midi=document["midi"])
    session.apply(
        {
            key: value
            for key, value in document.items()
            if key in ("channels", "drums", "notes", "drum_keys", "tuning")
        }
    )
    raw, stats = session.compile()
    data = json.loads(raw)

    # Sorted by time so the alternation follows what a listener actually hears,
    # not the order entities happen to appear in the document.
    events = sorted(_pitch_events(data), key=lambda item: item[0])
    for index, (_time_ms, call) in enumerate(events):
        call["args"]["item[1]"]["float"] = LOW_PITCH if index % 2 == 0 else HIGH_PITCH

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    print(f"rewrote {len(events)} pitch events, alternating {LOW_PITCH:+g}/{HIGH_PITCH:+g}")
    print(f"song notes: {stats['notes']}   emitters: {stats['peak_voices']}")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="the song's .snapmap.json sidecar")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "examples"
        / "song_pitch_probe.rawmap.json",
    )
    args = parser.parse_args()
    build(args.settings, args.output)


if __name__ == "__main__":
    main()
