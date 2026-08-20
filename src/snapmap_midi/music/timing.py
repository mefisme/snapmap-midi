"""The source clock: where the bar lines fall, and how fast the song is played.

The piano roll's ruler has to speak in bars and note values rather than
inventing evenly spaced seconds, so it needs the file's own tempo and
time-signature map. That map is read once, at import, and stored on the `Song`
-- the file it came from may be gone by the time a project is reopened, and
nothing about editing a note changes where a bar line falls.

`playback_speed` is the one lever that moves this. It is a playback multiplier
layered on top of the file's tempo map, not a rewrite of it, so it divides every
elapsed millisecond uniformly and leaves every tick alone.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction

#: How much of a bar past the last bar line is read as padding rather than as
#: music. Editors routinely write End-of-Track a handful of ticks past the final
#: bar, and a bare ceiling turns one stray tick into a whole empty measure on the
#: ruler -- 2,000 ms of dead timeline bought with 1/1920th of a bar. A 32nd note
#: is the smallest division the roll draws, so content shorter than that is not
#: content. A genuinely incomplete final measure is far larger than this and
#: still earns its bar.
_GRID_PADDING_BARS = Fraction(1, 32)

#: The manifest keys that are milliseconds and therefore move with speed.
_SCALED_KEYS = ("source_duration_ms", "grid_duration_ms")


def manifest(mid) -> dict:
    """The source MIDI clock as absolute tick/time change points, at speed 1.0.

    MIDI defines 120 BPM and 4/4 when a file omits the corresponding metadata.
    Markers at the same tick replace one another so a normal tick-zero tempo or
    signature does not leave a redundant default entry in the payload.

    Read at the file's own authored speed and never at a lever's. `at_speed`
    below applies the multiplier, which keeps the stored map a fact about the
    file rather than a reading of one setting at one moment.
    """
    import mido

    ticks_per_beat = int(mid.ticks_per_beat)
    tick = 0
    elapsed_s = 0.0
    tempo = 500_000
    tempos = [{"tick": 0, "time_ms": 0.0, "tempo": tempo}]
    signatures = [{"tick": 0, "time_ms": 0.0, "numerator": 4, "denominator": 4}]

    def record(markers, marker):
        if markers[-1]["tick"] == marker["tick"]:
            markers[-1] = marker
        else:
            markers.append(marker)

    for message in mido.merge_tracks(mid.tracks):
        delta = int(message.time)
        elapsed_s += mido.tick2second(delta, ticks_per_beat, tempo)
        tick += delta
        time_ms = round(elapsed_s * 1000, 6)
        if message.type == "set_tempo":
            tempo = int(message.tempo)
            record(tempos, {"tick": tick, "time_ms": time_ms, "tempo": tempo})
        elif message.type == "time_signature":
            record(
                signatures,
                {
                    "tick": tick,
                    "time_ms": time_ms,
                    "numerator": int(message.numerator),
                    "denominator": int(message.denominator),
                },
            )

    source_duration_ticks = tick
    signature = signatures[-1]
    ticks_per_bar = Fraction(
        ticks_per_beat * 4 * int(signature["numerator"]),
        int(signature["denominator"]),
    )
    ticks_since_signature = max(0, source_duration_ticks - int(signature["tick"]))
    raw_bars = Fraction(ticks_since_signature, 1) / ticks_per_bar
    completed_bars = math.ceil(raw_bars - _GRID_PADDING_BARS) if raw_bars else 0
    # Absorbing the padding must never absorb the music with it: any content at
    # all occupies at least the measure it started in.
    if ticks_since_signature and completed_bars < 1:
        completed_bars = 1
    grid_duration_ticks = max(
        source_duration_ticks,
        int(math.ceil(Fraction(int(signature["tick"]), 1) + completed_bars * ticks_per_bar)),
    )

    def time_at_tick(target_tick):
        marker = tempos[0]
        for candidate in tempos[1:]:
            if int(candidate["tick"]) > target_tick:
                break
            marker = candidate
        return round(
            float(marker["time_ms"])
            + (target_tick - int(marker["tick"])) * int(marker["tempo"]) / 1000 / ticks_per_beat,
            6,
        )

    return {
        "ticks_per_beat": ticks_per_beat,
        # Keep the file boundary and the workstation boundary separate. Some
        # DAWs write End-of-Track at the final note even when their clip still
        # has an empty remainder. The source value remains available for exact
        # MIDI accounting; the padded value gives the piano roll a complete
        # final measure without inventing another note or stop event.
        "duration_ticks": source_duration_ticks,
        "source_duration_ms": time_at_tick(source_duration_ticks),
        "grid_duration_ticks": grid_duration_ticks,
        "grid_duration_ms": time_at_tick(grid_duration_ticks),
        "tempo_changes": tempos,
        "time_signatures": signatures,
        # The file's own initial tempo, unaffected by `speed` -- that lever is a
        # playback multiplier layered on top, not a rewrite of what the file
        # says. MIDI's own default (120 BPM, `tempo = 500_000`) stands in when
        # the file never sets one, exactly as it does for playback itself.
        "base_bpm": round(60_000_000.0 / tempos[0]["tempo"], 2),
    }


def at_speed(stored: dict, speed: float) -> dict:
    """The stored clock as the transport will actually run it.

    Every millisecond divides and every tick stays, because speed compresses
    elapsed time without moving a single event in the file. `base_bpm` stays
    too: it is what the file says its tempo is, and auditioning a song at double
    speed does not rewrite that.
    """
    if not stored:
        return {}
    if speed == 1.0:
        return copy.deepcopy(stored)
    scaled = copy.deepcopy(stored)
    for key in _SCALED_KEYS:
        if scaled.get(key) is not None:
            scaled[key] = round(float(scaled[key]) / speed, 6)
    for markers in ("tempo_changes", "time_signatures"):
        for marker in scaled.get(markers, ()):
            marker["time_ms"] = round(float(marker["time_ms"]) / speed, 6)
    return scaled
