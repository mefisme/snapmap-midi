"""What the window is looking at: one song, one document, and one compile of them.

The song is HELD, not re-read. It is imported once, from a `.mid` or from a
saved project, and every compile and every preview reads it from memory
afterwards. That is the difference between a converter and an editor: a
converter can afford to re-parse its input, and an editor cannot, because an
edit that has not been written back to disk would not survive the next parse.

Every control in the window is a patch to a settings document, and that document
is projected onto the song's tracks and notes the moment it changes. Both live
here, in one object, so they cannot disagree: the rulers, the status line and the
exported bytes are three readings of the same state at the same moment rather
than three readings taken while somebody was still typing.

The state is behind a lock because pywebview answers each Javascript call on its
own thread. Two dropdowns changed in quick succession are two threads inside
`apply`, and an unguarded read-modify-write loses one of the changes with
nothing anywhere to show that it happened -- the control snaps back a beat later
and looks like a rendering glitch.

The palette's note index is built once, here, and handed to every compile.
Dropdowns fire immediately rather than on a debounce, so a re-derivation per
change is a re-derivation per click.

Nothing in here writes to the window and nothing imports pywebview. The bridge
above it turns these answers into `{"ok": ...}` payloads and catches what they
raise; keeping that split is what lets every rule in this file be tested with no
browser engine present.
"""

from __future__ import annotations

import copy
from pathlib import Path
from threading import RLock

from snapmap_midi import paths, project
from snapmap_midi import settings as settings_module
from snapmap_midi.compile import (
    compile_song,
    installed_event_duration_ms,
    installed_event_is_looping,
)
from snapmap_midi.music import analysis, importer, pipeline
from snapmap_midi.music import song as song_module
from snapmap_midi.music import timing as timing_module
from snapmap_midi.music.levers import compile_levers
from snapmap_midi.music.midi import for_part
from snapmap_midi.sound import palette
from snapmap_midi.ui.history import Command, History

#: The axis every ruler row is drawn against: MIDI's own range, which is also
#: the widest any family reaches. Fixed rather than derived from the file,
#: because rows are only comparable while they share one axis -- an axis taken
#: from the notes present would move when a channel was muted, and the same part
#: would sit somewhere else on the strip for a reason that has nothing to do
#: with it.
_AXIS = (0, 127)

#: How many bars a brand-new, drawn-from-nothing song starts with. There is no
#: file to read a length from -- `import_song` seeds `duration_ms` from the
#: MIDI file's own grid-completed measure, and a blank song has none -- so
#: this is a plain judgement call rather than a derived number: eight bars is
#: enough room to start composing without immediately reaching for "Set Song
#: Length...", and short enough that an empty roll does not read as broken.
_NEW_SONG_BARS = 8
_NEW_SONG_TICKS_PER_BEAT = 480
#: MIDI's own default tempo (120 BPM, 500,000 microseconds per beat) and time
#: signature (4/4) -- the same defaults `music/timing.py::manifest` falls
#: back to for a file that never sets either.
_NEW_SONG_TEMPO_US = 500_000


def _blank_timing() -> dict:
    """A `Song.timing` manifest for a song with no `.mid` behind it at all.

    `preview_manifest` reads `timing["source_duration_ms"]` unconditionally,
    and the ruler's own tick math (`ticksPerBar`/`timeAtTick` in `app.js`)
    reads `ticks_per_beat` and the tempo/signature markers -- an empty `{}`
    (what `Session._current_timing` already answers for a `None` song) would
    make either raise or draw nothing the moment `new_song` handed out a real
    song with nothing else populated. This is the same shape
    `music/timing.py::manifest` builds from a file, at the same MIDI
    defaults, for a song that has no file to build it from.
    """
    ticks_per_bar = _NEW_SONG_TICKS_PER_BEAT * 4
    duration_ticks = ticks_per_bar * _NEW_SONG_BARS
    duration_ms = round(duration_ticks * _NEW_SONG_TEMPO_US / 1000 / _NEW_SONG_TICKS_PER_BEAT, 6)
    return {
        "ticks_per_beat": _NEW_SONG_TICKS_PER_BEAT,
        "duration_ticks": 0,
        "source_duration_ms": 0.0,
        "grid_duration_ticks": duration_ticks,
        "grid_duration_ms": duration_ms,
        "tempo_changes": [{"tick": 0, "time_ms": 0.0, "tempo": _NEW_SONG_TEMPO_US}],
        "time_signatures": [{"tick": 0, "time_ms": 0.0, "numerator": 4, "denominator": 4}],
        "base_bpm": round(60_000_000.0 / _NEW_SONG_TEMPO_US, 2),
    }


def _percussion_modes(doc) -> tuple:
    """Every part's declared percussion mode, in a form two documents compare by.

    The analysis has to be re-read when one of these changes, for exactly the
    reason the drums switch forces a re-read: the mode decides whether a part is
    a kit, and a row still offering a family dropdown for a part the compiler
    has started routing through `DRUM_MAP` is the window describing an
    instrument nothing plays.
    """
    return tuple(
        sorted(
            (key, entry.get("percussion", "auto")) for key, entry in doc.get("channels", {}).items()
        )
    )


def _kb(size: int) -> str:
    """A byte count a musician can read. Rounded, because these are budgets."""
    if size >= 1024 * 1024:
        return "%.1f MB" % (size / (1024.0 * 1024.0))
    return "%d KB" % (size // 1024)


def _advice(destination: Path) -> str:
    """How to reach a map that has just been written to `destination`.

    The three cases are genuinely different and used to be two. Writing to the
    working directory because there is no loader folder is NOT "it landed where
    the loader reads", and saying so is the same quiet wrong answer the retired
    `--out` flag produced -- a file that looks finished and is not.
    """
    if paths.destination_is_loadable(destination):
        return "load it with `sh_rawmaps_on` in the console, then open any map"
    if paths.loader_dir() is None:
        return (
            "no loader folder on this platform -- copy it to the game machine's "
            "%%LOCALAPPDATA%%\\%s\\%s to play it" % (paths.LOADER_DIR_NAME, paths.RAWMAP_NAME)
        )
    return "the loader only reads %s; move it there to play it" % paths.rawmap_destination()


class Session:
    """One open song, the document that says what to do with it, and the numbers.

    Constructed with a song, a settings document, both, or neither -- the window
    opens on whatever the command line was given, including nothing at all.

    Opening a song at construction is not the same act as opening one later,
    and the difference is deliberate. `load` is the Open button: there is an
    earlier song whose instruments must not follow the user into the next file.
    The constructor has no earlier song, so it keeps everything the settings
    document brought with it -- otherwise `snapmap-midi ui song.mid --settings
    s.json`, which is one instruction, would throw away half of itself.
    """

    def __init__(self, midi=None, settings_path=None):
        """Open on a song, a settings document, or nothing.

        A song named here must be readable and raises if it is not: the caller
        just said which file, and a window that opened blank after being handed
        a path would be reporting nothing about the one thing it was told.

        A song REMEMBERED by a settings document is different. That path was
        recorded in an earlier session and the file may since have been renamed,
        moved, or handed to somebody else. The document is an afternoon's tuning
        and the path is one line of it, so a song that cannot be read leaves the
        document intact and the analysis empty. The guard is deliberately broad:
        what matters is that the settings survive, and a file can fail to be a
        MIDI file in as many ways as a file can be wrong.
        """
        self._lock = RLock()
        self._note_index = palette.build_note_index()
        self._analysis = None
        self._song = None
        self._history = History()
        if settings_path is None:
            self._doc = settings_module.defaults()
        else:
            self._doc = settings_module.load(settings_path)
        if midi is not None:
            self._doc = settings_module.merge(self._doc, {"midi": str(midi)})
        if self._doc["midi"]:
            try:
                self._adopt(project.open_midi(self._doc["midi"], self._doc))
            except Exception:
                if midi is not None:
                    raise

    # ---- the song ----

    def _adopt(self, song) -> None:
        """Take a song as the thing this session is editing from now on.

        The undo history is cleared with it. A command holds callables that
        close over the notes of the song it was recorded against, so replaying
        one onto a different song would mutate objects nothing is looking at any
        more -- and it is what any editor does when a different file is opened.
        """
        self._song = song
        self._analysis = self._analyze()
        self._history.clear()

    def _analyze(self):
        """Read the open song's parts, as the window's rows describe them.

        Derived from the song rather than from the file, so it says what is
        being edited rather than what was once imported. Both the drums mode and
        each track's percussion mode already live on the song, which is what
        keeps a row that draws a kit and a compile that routes it through
        `DRUM_MAP` from ever describing different arrangements.
        """
        return None if self._song is None else analysis.from_song(self._song)

    def song(self):
        """The song this session is editing, or None when nothing is open.

        Handed out live rather than copied: this is the mutable document, and
        the structural edits that arrive from Phase 2 onward change it in place
        under the lock this class holds.
        """
        with self._lock:
            return self._song

    def _current_timing(self) -> dict:
        """The song's tempo/time-signature map, as the transport will run it.

        The map itself was read once, at import, and nothing about editing a
        note moves a bar line. `playback_speed` is the one lever that reaches it,
        and it reaches it by division rather than by re-reading anything.
        """
        if self._song is None:
            return {}
        return timing_module.at_speed(self._song.timing, self._doc["tuning"]["playback_speed"])

    def load(self, midi_path) -> dict:
        """Open a song, forgetting the last one's instruments and keeping the setup.

        `channels` and `drum_keys` are cleared. They are answers about a
        particular arrangement, and channel numbers collide where parts do not:
        the marimba somebody chose for channel 3 would follow them into the next
        file and silently retimbre a part they have never looked at.

        `tuning`, `button` and `out_dir` stay. Those are about the user's setup
        rather than the song, and clearing them would make every new file a
        fresh argument with the same machine.

        The file is read before anything is stored, so a path that turns out not
        to be there leaves the session showing the song it already had rather
        than a half-opened one it never read.
        """
        with self._lock:
            midi_path = str(midi_path)
            candidate = dict(self._doc)
            candidate["midi"] = midi_path
            candidate["channels"] = {}
            candidate["notes"] = {}
            candidate["drum_keys"] = {}
            candidate = settings_module.validate(candidate)

            fresh = project.open_midi(midi_path, candidate)
            self._doc = candidate
            self._adopt(fresh)
            return analysis.as_dict(self._analysis)

    def analysis_dict(self) -> dict | None:
        """The open song as JSON, or None when there is no song open.

        None rather than an empty analysis: "no file yet" and "a file with
        nothing in it" are different states and the window says different things
        about them.
        """
        with self._lock:
            return None if self._analysis is None else analysis.as_dict(self._analysis)

    def channel_info(self, channel):
        """The analyzed part, named either by its key or by a bare channel.

        A part key -- "1:0" -- names exactly one part. A bare channel number is
        still accepted because it is what every existing caller sends and what a
        user means on the ordinary file where a channel holds one part; it
        resolves to the first part on that channel, which on such a file is the
        only one.
        """

        if self._analysis is None:
            raise ValueError("no song is open -- open a MIDI file first")

        info = None
        if isinstance(channel, str) and ":" in channel:
            info = next((c for c in self._analysis.channels if c.key == channel), None)
            if info is None:
                raise ValueError("no part %s in this song" % channel)
        else:
            if isinstance(channel, bool):
                raise ValueError("a MIDI channel from 0 to 15 is required")
            try:
                number = int(channel)
                valid = float(channel) == number and 0 <= number <= 15
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError("a MIDI channel from 0 to 15 is required")
            info = next((c for c in self._analysis.channels if c.channel == number), None)
            if info is None:
                raise ValueError("channel %d has no notes to anchor" % number)

        if info.lowest is None or info.highest is None:
            raise ValueError("%s has no notes to anchor" % info.key)
        return info

    # ---- editing notes ----

    def _track_and_note(self, track_id, note_id):
        """The track and note one structural edit names, or why neither exists.

        Notes have no flat index -- `Track.notes` is the only list holding
        them -- so every edit starts by walking the one track the caller says
        the note is on. `track_id` is asked for rather than searching every
        track for the note id: a stale `track_id` from a window that has not
        yet heard about a track being removed is a wrong edit, and this
        refuses it instead of quietly acting on whichever track still happens
        to hold that note id.
        """
        if self._song is None:
            raise ValueError("no song is open -- open a MIDI file first")
        track = self._song.track_by_id(track_id)
        if track is None:
            raise ValueError("no track %r in this song" % (track_id,))
        note = next((n for n in track.notes if n.id == note_id), None)
        if note is None:
            raise ValueError("no note %r on track %r" % (note_id, track_id))
        return track, note

    def _track_only(self, track_id):
        """The track one structural edit names, or why it doesn't exist.

        The track-only half of `_track_and_note`'s lookup, for an edit that
        does not name an existing note: drawing a new one, deleting a track,
        renaming it, or reopening it from its source file.
        """
        if self._song is None:
            raise ValueError("no song is open -- open a MIDI file first")
        track = self._song.track_by_id(track_id)
        if track is None:
            raise ValueError("no track %r in this song" % (track_id,))
        return track

    @staticmethod
    def _require_midi_value(value, what: str) -> int:
        """A MIDI-range integer (0-127): a written pitch or a velocity.

        A bool is refused explicitly -- `isinstance(True, int)` is true in
        Python, so `False` would otherwise pass as pitch 0 and nobody asked
        for that note.
        """
        if isinstance(value, bool):
            raise ValueError("%s is %r; it has to be a MIDI value from 0 to 127" % (what, value))
        try:
            number = int(value)
            exact = float(value) == number
        except (TypeError, ValueError):
            exact = False
        if not exact or not (0 <= number <= 127):
            raise ValueError("%s is %r; it has to be a MIDI value from 0 to 127" % (what, value))
        return number

    @staticmethod
    def _require_ms(value, what: str, *, minimum: int) -> int:
        """A whole number of milliseconds, at or above `minimum`."""
        if isinstance(value, bool):
            raise ValueError(
                "%s is %r; it has to be a whole number of milliseconds" % (what, value)
            )
        try:
            number = int(value)
            exact = float(value) == number
        except (TypeError, ValueError):
            exact = False
        if not exact or number < minimum:
            raise ValueError(
                "%s is %r; it has to be a whole number of milliseconds, %d or more"
                % (what, value, minimum)
            )
        return number

    def _fit_length(self, new_end_ms: int) -> tuple:
        """Grow the song to cover a note end, never shrink it. Returns `(before, after)`.

        The song used to have no stored length at all -- it was recomputed
        live from the furthest note every time, so dragging a note past the
        edge just worked, the visible song grew to fit. `Song.duration_ms`
        being authoritative now must not regress that: a note dragged or
        resized past the current length still has to stay audible and
        undimmed, so the length grows to cover it automatically. Only growth
        is automatic -- shrinking is `set_song_length`'s alone to do, which is
        why this never returns a value smaller than what is already stored.
        """
        before = self._song.duration_ms
        return before, max(before, new_end_ms)

    def move_note(self, track_id, note_id, start_ms, pitch) -> None:
        """Move a note to a new start time and written pitch.

        Both move together because that is what a body-drag on the roll IS --
        the roll never asks for one without the other -- and one undo step
        should put both back rather than leaving a half-dragged note one
        Ctrl+Z away from where it actually started. The song's own length
        grows with it if the note now reaches past the end (see `_fit_length`);
        the growth rides in the same undo step so Ctrl+Z restores both.
        """
        with self._lock:
            song = self._song
            track, note = self._track_and_note(track_id, note_id)
            start_ms = self._require_ms(start_ms, "start_ms", minimum=0)
            pitch = self._require_midi_value(pitch, "pitch")
            before = (note.start_ms, note.pitch)
            after = (start_ms, pitch)
            duration_before, duration_after = self._fit_length(start_ms + note.duration_ms)

            def _apply():
                note.start_ms, note.pitch = after
                song.duration_ms = duration_after

            def _revert():
                note.start_ms, note.pitch = before
                song.duration_ms = duration_before

            _apply()
            self.push_command("Move note", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    def resize_note_start(self, track_id, note_id, start_ms) -> None:
        """Move a note's start, keeping its end fixed instead of its duration.

        The front-edge drag: dragging a note's LEFT edge should anchor the
        note's end and let duration follow, which is the opposite of what
        `move_note` does. Asking for a duration here would make the caller
        compute it from the current end anyway, so this takes the one number
        a front-edge drag actually produces -- where the edge landed.
        """
        with self._lock:
            track, note = self._track_and_note(track_id, note_id)
            end_ms = note.start_ms + note.duration_ms
            start_ms = self._require_ms(start_ms, "start_ms", minimum=0)
            if start_ms >= end_ms:
                raise ValueError(
                    "start_ms %r is not before this note's end at %d ms" % (start_ms, end_ms)
                )
            before = (note.start_ms, note.duration_ms)
            after = (start_ms, end_ms - start_ms)

            def _apply():
                note.start_ms, note.duration_ms = after

            def _revert():
                note.start_ms, note.duration_ms = before

            _apply()
            self.push_command("Resize note", revert=_revert, apply=_apply)

    def resize_note(self, track_id, note_id, duration_ms) -> None:
        """Change a note's written duration, keeping its start and pitch.

        Duration alone: dragging a note's right edge never moves its start, so
        this asks for exactly the one number that gesture produces. Duration
        does not move a note's pitch or its position in a channel's pitch
        histogram, so unlike `move_note` and `delete_note` this used not to
        re-derive the analysis -- it does now, because stretching a note past
        the song's own length can grow that length (see `_fit_length`), and
        `duration_s` in the analysis is read straight off it.
        """
        with self._lock:
            song = self._song
            track, note = self._track_and_note(track_id, note_id)
            duration_ms = self._require_ms(duration_ms, "duration_ms", minimum=1)
            before = note.duration_ms
            duration_before, duration_after = self._fit_length(note.start_ms + duration_ms)

            def _apply():
                note.duration_ms = duration_ms
                song.duration_ms = duration_after

            def _revert():
                note.duration_ms = before
                song.duration_ms = duration_before

            _apply()
            self.push_command("Resize note", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    def delete_note(self, track_id, note_id) -> None:
        """Remove a note from its track, and remember exactly where it sat.

        `index` is captured before the removal so undo reinserts the note at
        the same position in `track.notes` rather than at the end. Nothing
        downstream reads that position as meaning -- the compiler and the
        preview both re-sort before a byte is written or a note is scheduled
        -- but putting a restored note back exactly where it was is still the
        more honest undo of the two, and it is free to do.
        """
        with self._lock:
            track, note = self._track_and_note(track_id, note_id)
            index = track.notes.index(note)

            def _apply():
                if note in track.notes:
                    track.notes.remove(note)

            def _revert():
                if note not in track.notes:
                    track.notes.insert(min(index, len(track.notes)), note)

            _apply()
            self.push_command("Delete note", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    def set_note_velocity(self, track_id, note_id, velocity) -> None:
        """Change a note's written MIDI velocity.

        Velocity feeds the note's baseline loudness (`expression.midi_velocity_db`);
        it is a fact about how hard the note was struck, not the same thing as
        the Note volume override the inspector already exposes, which replaces
        that baseline rather than describing it. Like `resize_note`, this does
        not touch a channel's pitch histogram, so the analysis is left alone.
        """
        with self._lock:
            track, note = self._track_and_note(track_id, note_id)
            velocity = self._require_midi_value(velocity, "velocity")
            before = note.velocity

            def _apply():
                note.velocity = velocity

            def _revert():
                note.velocity = before

            _apply()
            self.push_command("Change note velocity", revert=_revert, apply=_apply)

    def create_note(self, track_id, pitch, start_ms, duration_ms, velocity=100) -> None:
        """Draw a new note onto an existing track.

        Mints its id through `Song.new_note_id`, exactly like every note an
        import creates -- a drawn note is not a different kind of note, only
        a different way of arriving. The song's own length grows to cover it
        if it lands past the current end, in the same undo step, the same as
        `move_note`/`resize_note` already do for a dragged note.
        """
        with self._lock:
            song = self._song
            track = self._track_only(track_id)
            pitch = self._require_midi_value(pitch, "pitch")
            velocity = self._require_midi_value(velocity, "velocity")
            start_ms = self._require_ms(start_ms, "start_ms", minimum=0)
            duration_ms = self._require_ms(duration_ms, "duration_ms", minimum=1)
            note = song_module.Note(
                id=song.new_note_id(),
                pitch=pitch,
                velocity=velocity,
                start_ms=start_ms,
                duration_ms=duration_ms,
            )
            duration_before, duration_after = self._fit_length(start_ms + duration_ms)

            def _apply():
                if note not in track.notes:
                    track.notes.append(note)
                song.duration_ms = duration_after

            def _revert():
                if note in track.notes:
                    track.notes.remove(note)
                song.duration_ms = duration_before

            _apply()
            self.push_command("Draw note", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    # ---- tracks ----

    def create_track(self, name="") -> str:
        """Add a blank track with no notes and no MIDI provenance. Returns its id.

        `source_track=-1` and `source_midi=None` are the convention `Track`'s
        own docstring establishes for a track drawn from nothing. `channel`
        cannot be 0 (or any other fixed value) for every hand-drawn track:
        `Track.key` is `"%d:%d" % (source_track, channel)`, and every
        hand-drawn track shares `source_track == -1`, so `channel` is the only
        thing left to keep two of them from colliding in the settings
        document that `project.to_settings` keys by that same string. Minted
        through `Song.new_drawn_channel` rather than derived from the tracks
        currently in the song -- see that method's own docstring for why a
        deleted track's channel must never come back into use.

        Deliberately does not set `sound`/`family`/anything else `Track`
        carries: `project.apply_settings` calls `track.clear_levers()` and
        re-derives every lever from the settings document on EVERY patch
        application, for EVERY track, so a value set directly here would be
        silently wiped the next time the user touched any unrelated control.
        Assigning a sound is the same `openSoundBrowser` -> pick -> `apply_settings`
        path an imported track already uses, triggered by the window right
        after this returns -- not this method's job.
        """
        with self._lock:
            song = self._song
            if song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            if not isinstance(name, str):
                raise ValueError("a track name has to be text")
            track_id = song.new_track_id()
            track = song_module.Track(
                id=track_id,
                name=name,
                channel=song.new_drawn_channel(),
                source_track=-1,
                source_midi=None,
                notes=[],
            )

            def _apply():
                if track not in song.tracks:
                    song.tracks.append(track)

            def _revert():
                if track in song.tracks:
                    song.tracks.remove(track)

            _apply()
            self.push_command("Create track", revert=_revert, apply=_apply)
            self._analysis = self._analyze()
            return track_id

    def delete_track(self, track_id) -> None:
        """Remove a track, and every note on it, from the song.

        Undo-tracked exactly like `delete_note`: the index is captured so a
        restored track goes back where it was rather than at the end.

        A deleted track's `doc["channels"][track.key]` entry is left behind
        in the settings document -- nothing here or in `apply_settings`
        removes it, the same as an imported track's entry outlives that track
        being muted or re-pointed. That is only safe because `channel` (the
        one part of a hand-drawn track's key that can vary) is minted from
        `Song.new_drawn_channel`, a counter that never runs backward: no
        LATER track, hand-drawn or not, can ever be assigned the deleted
        track's channel number again, so nothing can ever address that stale
        entry again either. A track imported from a `.mid` cannot collide
        this way in the first place -- its key's `source_track` half is
        offset past every part already present at import time (see
        `music/importer.py::import_tracks`), so two different files' tracks
        never share a key even before this rule is considered.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            song = self._song
            track = self._track_only(track_id)
            index = song.tracks.index(track)

            def _apply():
                if track in song.tracks:
                    song.tracks.remove(track)

            def _revert():
                if track not in song.tracks:
                    song.tracks.insert(min(index, len(song.tracks)), track)

            _apply()
            self.push_command("Delete track", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    def rename_track(self, track_id, name) -> None:
        """Change a track's display name. Undo-tracked like any other edit."""
        with self._lock:
            track = self._track_only(track_id)
            if not isinstance(name, str):
                raise ValueError("a track name has to be text")
            before = track.name

            def _apply():
                track.name = name

            def _revert():
                track.name = before

            _apply()
            self.push_command("Rename track", revert=_revert, apply=_apply)
            # `track_name` rides in the analysis (`ChannelInfo.track_name`),
            # not the settings document, so the row's label needs a fresh
            # analysis the same way a note edit that changes the pitch
            # histogram does.
            self._analysis = self._analyze()

    def reopen_track(self, track_id) -> None:
        """Re-read one track's notes from its original file. Keeps its levers and id.

        The per-track replacement for the old, whole-song "Reopen MIDI":
        notes refresh, every lever on the track (sound, mute, transpose, ...)
        stays exactly as the user left it. Only available on a track that
        carries a `source_midi` -- a hand-drawn track has nothing to re-read.

        Matching which part of the file is "this track" again is not as
        simple as re-reading `track.source_track`: a fresh, standalone
        `importer.import_tracks(track.source_midi)` call numbers parts from
        that file's own raw indices (its internal offset is 0), which will
        not generally equal what is stored on `track.source_track` -- that
        value may have been offset at ORIGINAL import time to avoid colliding
        with a different file's tracks already in this project (see
        `import_tracks`'s `track_offset`). So this matches by POSITION within
        the file instead: every track in this song sharing the same
        `source_midi`, sorted by `(source_track, channel)`, is a stable
        ordering (offsetting every one of them by the same constant at import
        time preserves their relative order), and this track's index in that
        ordering is the same index a fresh import of the same file produces,
        sorted the same way. Only holds for the ordinary case -- the file's
        track/channel structure unchanged since import. If the file has
        genuinely changed shape since then, this may match the wrong part; a
        wrong-but-plausible match is treated as an accepted limitation here
        rather than a case this tries to detect or refuse.
        """
        with self._lock:
            song = self._song
            track = self._track_only(track_id)
            if track.source_midi is None:
                raise ValueError("%r has no source file to reopen" % (track.name or track_id))
            same_source = sorted(
                (t for t in song.tracks if t.source_midi == track.source_midi),
                key=lambda t: (t.source_track, t.channel),
            )
            position = same_source.index(track)
            fresh = importer.import_tracks(track.source_midi)
            fresh_sorted = sorted(fresh.tracks, key=lambda t: (t.source_track, t.channel))
            if position >= len(fresh_sorted):
                raise ValueError(
                    "%s no longer has a matching part in %s"
                    % (track.name or track.key, track.source_midi)
                )
            new_notes = fresh_sorted[position].notes
            before = track.notes
            new_end = max((n.start_ms + n.duration_ms for n in new_notes), default=0)
            duration_before, duration_after = self._fit_length(new_end)

            def _apply():
                track.notes = new_notes
                song.duration_ms = duration_after

            def _revert():
                track.notes = before
                song.duration_ms = duration_before

            _apply()
            self.push_command("Reopen from source", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    # ---- song length and loop ----

    def set_song_length(self, duration_ms) -> None:
        """Change how long the song is. Never deletes or clips a note.

        A note past the new length simply stops playing and exporting -- the
        same exclusion a mute already produces (see `beyond_length` in
        `music/midi.py::resolve_notes`) -- and starts playing again the moment
        the length grows back past it. Shrinking is therefore free to undo:
        nothing here ever touches `Track.notes`.

        The loop brace rides along if it would otherwise land past the new,
        shorter end: `set_loop` refuses a loop past the song's length, so
        leaving the brace where it was would strand it somewhere a later,
        unrelated loop edit could reject for a reason that has nothing to do
        with what that edit actually asked for. Growing the song never moves
        the brace -- only a shrink past its current edge does.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            song = self._song
            duration_ms = self._require_ms(duration_ms, "duration_ms", minimum=1)
            before = (song.duration_ms, song.loop_start_ms, song.loop_end_ms)
            loop_end = min(song.loop_end_ms, duration_ms)
            loop_start = min(song.loop_start_ms, max(0, loop_end - 1))
            after = (duration_ms, loop_start, loop_end)

            def _apply():
                song.duration_ms, song.loop_start_ms, song.loop_end_ms = after

            def _revert():
                song.duration_ms, song.loop_start_ms, song.loop_end_ms = before

            _apply()
            self.push_command("Set song length", revert=_revert, apply=_apply)
            self._analysis = self._analyze()

    def fit_song_length(self) -> None:
        """Set the song's length back to its content, in one undoable step.

        The same value the workstation always showed before `duration_ms`
        became a stored, editable fact: the later of the file's own
        grid-completed measure and the furthest note actually written, walked
        fresh across every track rather than read from import-time state, so
        this still gives the right answer after notes have been moved,
        resized, or deleted since. Delegates to `set_song_length` rather than
        duplicating its undo/loop-clamp/analysis handling.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            song = self._song
            content_ms = int(round((song.timing or {}).get("grid_duration_ms") or 0))
            for track in song.tracks:
                for note in track.notes:
                    content_ms = max(content_ms, note.start_ms + note.duration_ms)
            content_ms = max(1, content_ms)
        self.set_song_length(content_ms)

    def set_loop(self, start_ms, end_ms) -> None:
        """Move the loop brace to a new region, both edges at once.

        One undo entry for the pair, the same reason `move_note` sets `start`
        and `pitch` together: a body-drag on the brace, or a single
        edge-drag, is one gesture, and Ctrl+Z should put the WHOLE thing back
        rather than leaving one edge one step behind the other.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            song = self._song
            start_ms = self._require_ms(start_ms, "start_ms", minimum=0)
            end_ms = self._require_ms(end_ms, "end_ms", minimum=1)
            if start_ms >= end_ms:
                raise ValueError("loop start_ms %r is not before end_ms %r" % (start_ms, end_ms))
            if end_ms > song.duration_ms:
                raise ValueError(
                    "loop end_ms %r is past the song's length of %d ms" % (end_ms, song.duration_ms)
                )
            before = (song.loop_start_ms, song.loop_end_ms)
            after = (start_ms, end_ms)

            def _apply():
                song.loop_start_ms, song.loop_end_ms = after

            def _revert():
                song.loop_start_ms, song.loop_end_ms = before

            _apply()
            self.push_command("Move loop", revert=_revert, apply=_apply)

    def set_loop_enabled(self, enabled) -> None:
        """Turn the transport's "Loop playback" wrap on or off.

        Not undo-tracked, the same as mute and solo: it is a playback switch,
        not an edit to the song, and Ctrl+Z after toggling it should undo
        whatever note edit came before, not silently flip it back.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            if not isinstance(enabled, bool):
                raise ValueError("enabled has to be true or false")
            self._song.loop_enabled = enabled

    def export_loop(self) -> dict:
        """Write just the loop region as its own map, and say where it went.

        A pure windowing transform (`music/song.py::loop_window`) runs on a
        throwaway copy of the song first, so this never disturbs the song the
        workstation has open -- the same reason `compile`/`export` above are
        safe to call from a running window. `export()`'s file-write and
        `replaced` bookkeeping is repeated rather than shared with a flag,
        because the two now compile two different songs and reusing one
        method for both would mean threading a second `Song` through
        `compile()`'s own signature for a single caller.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            song = self._song
            if song.loop_start_ms >= song.loop_end_ms:
                raise ValueError(
                    "the loop region is empty (start_ms %r is not before end_ms %r)"
                    % (song.loop_start_ms, song.loop_end_ms)
                )
            windowed = song_module.loop_window(song, song.loop_start_ms, song.loop_end_ms)
            raw, stats = compile_song(
                windowed,
                self._baseline_bytes(),
                levers=compile_levers(windowed),
                note_index=self._note_index,
            )
            destination = paths.rawmap_destination(self._doc["out_dir"])
            replaced = destination.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            return {
                "destination": str(destination),
                "advice": _advice(destination),
                "replaced": replaced,
                "stats": self._report(stats),
            }

    # ---- the document ----

    def settings(self) -> dict:
        """The whole document, as a copy.

        A copy because the bridge hands this straight to Javascript and because
        it is the state every other method reads. A caller that edited the
        session's own dict would change what the next compile does without
        passing through `validate`, and sharing one mutable document across
        threads defeats the lock that guards it.
        """
        with self._lock:
            return copy.deepcopy(self._doc)

    def apply(self, patch) -> dict:
        """Merge a patch into the document, and project it onto the song.

        Validation runs before anything is stored, so a refused patch leaves the
        session exactly as it was. That matters more here than in a file: a
        half-applied patch is a window describing settings nobody chose, and the
        song underneath it playing a third thing.

        The parts are described again when the drums mode changes, because that
        switch decides whether channel 9 is a kit. Without it the row goes on
        offering a family dropdown for a channel the compiler has started
        routing through `DRUM_MAP`. The song path is watched for the same
        reason, though the window changes that through `load`.
        """
        with self._lock:
            merged = settings_module.merge(self._doc, patch)
            reopened = merged["midi"] != self._doc["midi"]
            reread = (merged["drums"], _percussion_modes(merged)) != (
                self._doc["drums"],
                _percussion_modes(self._doc),
            )
            previous = self._doc
            self._doc = merged
            if reopened:
                # `load` is the door for this, and the window only ever uses
                # that one. A patch can still name a different song -- the
                # command line's `--settings` remembers one -- and a session
                # holding the old song's notes under the new song's name would
                # export music from a file nobody has open.
                try:
                    self._adopt(project.open_midi(merged["midi"], merged))
                except Exception:
                    self._doc = previous
                    raise
            elif self._song is not None:
                project.apply_settings(self._song, merged)
                if reread:
                    self._analysis = self._analyze()
            return copy.deepcopy(merged)

    def reanalyze(self) -> None:
        """Describe the song's parts again with nothing in the document changed.

        The percussion table is a preference, not a setting of this song, so
        it lives outside the document and saving one moves what `drum_keys`
        falls back to without any patch for `apply` to notice. Without this
        the window goes on drawing -- and the preview goes on playing -- the
        old kick until the song is reopened.
        """
        with self._lock:
            if self._song is not None:
                self._analysis = self._analyze()

    # ---- undo ----

    def push_command(self, label: str, revert, apply=None) -> None:
        """Record a change already made to the song, and how to take it back.

        Nothing in this phase calls it: no structural edit exists yet. It is
        here now so the ones that arrive next -- move, resize, delete, draw, add
        a track -- push onto a stack that already works rather than one designed
        around whichever of them happens to be written first.
        """
        with self._lock:
            self._history.push(Command(label=label, apply=apply or (lambda: None), revert=revert))

    def undo(self) -> dict:
        """Take back the last change, and say what is left to take back."""
        with self._lock:
            label = self._history.undo()
            if label is not None:
                self._analysis = self._analyze()
            state = self._history.state()
            state["undone"] = label
            return state

    def redo(self) -> dict:
        """Put back the last undone change."""
        with self._lock:
            label = self._history.redo()
            if label is not None:
                self._analysis = self._analyze()
            state = self._history.state()
            state["redone"] = label
            return state

    def history_state(self) -> dict:
        with self._lock:
            return self._history.state()

    # ---- starting or growing a project ----

    def new_song(self) -> dict:
        """Start a blank song: no tracks, no `.mid`, nothing chosen about it yet.

        The entry point that makes composing from nothing reachable at all --
        every other way into this session starts from a `.mid`. Seeded the
        same way `import_song` seeds an imported one (`music/importer.py`):
        the loop brace spans the whole song and `duration_ms` is a real,
        positive number, both from `_blank_timing`'s made-up-but-honest
        manifest rather than the `{}` `Session._current_timing` already
        answers for "no song at all" -- `preview_manifest` reads
        `timing["source_duration_ms"]` unconditionally, so a truly empty
        manifest would fail the first redraw of a new song rather than
        showing one.
        """
        with self._lock:
            timing = _blank_timing()
            duration_ms = max(1, int(round(timing["grid_duration_ms"])))
            song = song_module.Song(
                duration_ms=duration_ms,
                loop_start_ms=0,
                loop_end_ms=duration_ms,
                timing=timing,
                tracks=[],
            )
            self._doc = settings_module.defaults()
            self._adopt(song)
            return analysis.as_dict(self._analysis)

    def import_midi_into_project(self, mid_path) -> dict:
        """Add another `.mid` to the OPEN project, as further tracks beside what
        is already there -- rather than replacing it, which is what `load` does.

        `music.importer.add_midi` is already exactly this operation (composable
        by design since Phase 1: fresh, collision-free track/note ids, offsets
        clear of every part already present). It mutates `song.tracks`,
        `song.duration_ms` and `song.timing` in place and hands back the
        `Track` objects it added, which is what undo needs to know to take
        only them back out again.

        `project.apply_settings(self._song, self._doc)` afterward is safe to
        call on the whole song, not just the new tracks: every EXISTING
        track's entry is already keyed in `self._doc["channels"]` by its own
        `Track.key`, so re-running the projection on them reproduces the
        exact values they already had -- this is the same call `Session.apply`
        already makes on every settings patch, for every track, every time.
        The newly added tracks have no entry yet, so they come back with
        every lever cleared (`Track.clear_levers`), same as `create_track`.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            song = self._song
            added = importer.add_midi(song, mid_path)
            project.apply_settings(song, self._doc)

            def _apply():
                for track in added:
                    if track not in song.tracks:
                        song.tracks.append(track)

            def _revert():
                for track in added:
                    if track in song.tracks:
                        song.tracks.remove(track)

            self.push_command("Import MIDI", revert=_revert, apply=_apply)
            self._analysis = self._analyze()
            return analysis.as_dict(self._analysis)

    # ---- the project file ----

    def project_destination(self):
        """Where this song's project file would be written, or None.

        Beside the `.mid`, named after it -- the same convention the settings
        sidecar uses, and for the same reason: a dialog would mean a file the
        user has to keep track of, while the song is the thing they already have
        open.
        """
        with self._lock:
            song = self._song
            if song is None or not song.origin:
                return None
            return project.project_path(song.origin)

    def save_project(self, path=None):
        """Write the song to a project file, and say where it went.

        The `.mid` is never touched. An import is a starting point, and a
        starting point that gets overwritten is not one.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            if path is None:
                path = self.project_destination()
            if path is None:
                raise ValueError("this song has no file to sit beside -- choose where to save it")
            return project.save(self._song, path)

    def load_project(self, path) -> dict:
        """Open a saved project, replacing whatever this session was editing.

        The settings document is rebuilt from the song, because that document is
        what every control in the window patches. Only the keys about the user's
        setup -- where maps are written, which baseline to add to -- survive from
        the session that was open, since those are not part of the project.
        """
        with self._lock:
            song = project.load(path)
            self._doc = project.to_settings(song, self._doc)
            self._adopt(song)
            return analysis.as_dict(self._analysis)

    # ---- compiling ----

    def _baseline_bytes(self) -> bytes | None:
        """The saved map to add the song to, or None to author a blank one.

        An explicitly chosen baseline wins over a configured one, which is the
        same order the command line uses. Neither is the ordinary case.
        """
        chosen = self._doc["baseline"]
        path = Path(chosen) if chosen else paths.baseline_map()
        return path.read_bytes() if path else None

    def _levers(self) -> dict:
        """Every conversion choice, read off the song rather than the document.

        The document is where the choices are MADE; the song is where they are
        kept once a wildcard has been resolved onto the tracks it covered. Both
        say the same thing, and reading the song is what makes an edit that
        never touched the document -- a note moved, a track added -- reach the
        compile at all.
        """
        return compile_levers(self._song)

    def compile(self):
        """Compile the song as it stands: finished bytes and a statistics summary.

        Held under the lock for its whole duration rather than compiled from a
        snapshot. A compile that read the family list before a change and the
        mute set after it would produce a map matching neither, and the report
        printed beside it would describe a third thing. Serialising is the
        cheaper mistake: the window's Javascript already stamps each call it
        makes and drops answers that arrive out of order.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            return compile_song(
                self._song,
                self._baseline_bytes(),
                levers=self._levers(),
                note_index=self._note_index,
            )

    def stats(self) -> dict:
        """A dry run: what a compile of this song produces, plus the warnings.

        Really compiles. An estimate that disagreed with the export would be
        discovered in game, and closing that loop is the entire reason this
        window exists.
        """
        with self._lock:
            _, stats = self.compile()
            return self._report(stats)

    def preview_manifest(self, *, with_stats: bool = False):
        """The converted notes the workstation transport will actually play.

        The same conversion the export runs, stopping before the map is
        authored. Channel choices, duration caps, polyphony thinning and
        per-layer voice stealing all come from `music/pipeline.py`, which the
        compiler uses too -- so the preview is a reading of the current
        conversion rather than a General MIDI render, and it cannot drift from
        what gets exported.

        ``with_stats`` is the fast status path used by an interactive settings
        update. Authoring and serialising a complete map just to repaint a mute
        button makes a dense song feel frozen; the exact, export-ready report
        remains available through :meth:`stats` and ``dry_run``.
        """
        with self._lock:
            if self._song is None:
                raise ValueError("no song is open -- open a MIDI file first")
            levers = self._levers()
            notes, source_stats = pipeline.resolve(
                self._song,
                levers,
                note_index=self._note_index,
                include_silent=True,
                event_is_looping=installed_event_is_looping,
            )
            audible_notes = [note for note in notes if note.audible]
            prepared = pipeline.prepare(
                audible_notes, levers, duration_lookup=installed_event_duration_ms
            )
            scheduled_notes = list(prepared.shared_decaying) + list(prepared.isolated)
            scheduled_ids = {id(note) for note in scheduled_notes}

            def _track_id_for(note):
                """The `Track.id` a resolved note came from, or None.

                A resolved note only carries `track`/`chan` (the MIDI
                identity `pipeline.py` duck-types on), because the export and
                preview core stays MIDI-agnostic. This is the one place that
                identity is turned back into the song's own track, and only
                for the window -- nothing downstream of `pipeline.resolve`
                needs it.
                """
                owner = self._song.track_for_part(getattr(note, "track", 0), note.chan)
                return owner.id if owner is not None else None

            def _event_payload(note, *, converted):
                midi_end = max(note.start, getattr(note, "midi_end", note.end))
                sustain_limit_visual_end = getattr(note, "sustain_limit_visual_end", None)
                shortened_by = getattr(
                    note,
                    "shortened_by",
                    "sustain" if sustain_limit_visual_end is not None else None,
                )
                if shortened_by == "voices":
                    # A voice-stolen cut is a real-time fact about this
                    # compile, not an editorial setting -- draw it exactly
                    # where playback actually stops.
                    visual_end = max(note.start, getattr(note, "preview_end", note.end))
                elif sustain_limit_visual_end is not None:
                    # A Sustain Limit's cut is drawn at the SAME point
                    # regardless of the Track transpose currently being
                    # auditioned (see the untransposed comparison this reads
                    # in voices.py). Playback below keeps using the real,
                    # pitch-aware stop so audio is unaffected.
                    visual_end = min(midi_end, sustain_limit_visual_end)
                else:
                    visual_end = midi_end
                return {
                    "id": note.id,
                    "start": note.start,
                    # `end` is when the note stops being heard, after caps and
                    # emitter stealing -- the preview transport schedules from
                    # it. `midi_end` is what the file wrote, which is what the
                    # roll draws, so moving a tuning lever changes the shading
                    # on a block rather than the block. `visual_end` is where
                    # that shading itself is drawn -- pitch-stable for a
                    # Sustain Limit, real-time for a voice steal.
                    "end": max(
                        note.start,
                        getattr(note, "preview_end", note.end),
                    ),
                    "midi_end": midi_end,
                    "visual_end": visual_end,
                    "sound": note.shader,
                    "channel": note.chan,
                    # The part this note belongs to, matching `ChannelInfo.key`.
                    # Without it the roll sees only a channel, so three parts
                    # written to channel 0 are one undifferentiated mass and
                    # nothing can select, dim, or mute one of them.
                    "part": "%d:%d" % (getattr(note, "track", 0), note.chan),
                    "track": getattr(note, "track", 0),
                    # The stable `Track.id` this note lives on, distinct from
                    # `part`/`track` above: those name a MIDI identity
                    # (source track + channel), while `move_note`/
                    # `resize_note`/`delete_note`/`set_note_velocity` need the
                    # song's own track identity to find the note at all --
                    # `Track.notes` is the only place a note is stored, with
                    # no flat index across the song.
                    "track_id": _track_id_for(note),
                    "source_pitch": note.source_pitch,
                    "pitch": note.source_pitch,
                    "velocity": note.velocity,
                    "family": note.fam,
                    "sustained": note.sustained,
                    # A capped one-shot needs an explicit browser stop just
                    # like a stolen voice. Looping notes already use `end` as
                    # their release point in the preview scheduler.
                    "cut": bool(getattr(note, "preview_cut", False))
                    or bool(getattr(note, "sustain_limited", False) and not note.sustained),
                    # Which limit silenced this note, or None if none did. The
                    # roll draws it dimmed either way; the panel needs to know
                    # whose fault it was to report each lever separately.
                    "limited_by": getattr(note, "limited_by", None),
                    "shortened_by": shortened_by,
                    "audible": bool(note.audible),
                    "muted": bool(note.muted),
                    "solo_excluded": bool(note.solo_excluded),
                    "out_of_key_range": bool(getattr(note, "out_of_key_range", False)),
                    # A note whose start falls at or past the song's own
                    # length (see `Song.duration_ms`). Folded into `audible`
                    # the same way muting is, so the roll dims it through the
                    # same path a muted note already uses -- nothing new to
                    # draw, only a new reason a note can be excluded.
                    "beyond_length": bool(getattr(note, "beyond_length", False)),
                    "converted": bool(converted),
                    "pitch_follow": note.pitch_follow,
                    "root_pitch": note.profile_root_pitch,
                    "applied_root_pitch": note.root_pitch,
                    "root_confidence": note.root_confidence,
                    "root_source": note.root_source,
                    "pitch_offset": note.pitch_offset,
                    "pitch_semitones": note.pitch_semitones,
                    "track_transpose": note.track_transpose,
                    "fine_tune_cents": note.fine_tune_cents,
                    "manual_pitch_semitones": note.manual_pitch_semitones,
                    "follow_pitch_semitones": note.follow_pitch_semitones,
                    "automatic_pitch": note.automatic_pitch,
                    "requested_pitch": note.requested_pitch,
                    "pitch_modifier": note.pitch_modifier,
                    "glide_ms": int(getattr(note, "glide_ms", 0) or 0),
                    "glide_from_pitch": getattr(note, "glide_from_pitch", None),
                    "attack_ms": int(
                        for_part(
                            levers["part_attack_ms"],
                            getattr(note, "track", 0),
                            note.chan,
                            0,
                        )
                        or 0
                    ),
                    "release_s": for_part(
                        levers["part_release_s"],
                        getattr(note, "track", 0),
                        note.chan,
                        levers["release_s"],
                    ),
                    "hard_stop": bool(
                        for_part(
                            levers["part_hard_stop"],
                            getattr(note, "track", 0),
                            note.chan,
                            levers["hard_stop"],
                        )
                    ),
                    "pitch_limited": note.pitch_limited,
                    # The octave the part was folded into, automatic or pinned.
                    # Sent per note rather than recomputed in the browser so the
                    # window can never disagree with the exporter about where a
                    # track sits.
                    "octave_shift": getattr(note, "octave_shift", 0),
                    "playback_rate": note.playback_rate,
                    "velocity_db": note.velocity_db,
                    "volume_trim_db": note.volume_trim_db,
                    "note_volume_db": note.note_volume_db,
                    "track_volume_db": note.track_volume_db,
                    "master_volume_db": note.master_volume_db,
                    "requested_volume_db": note.requested_volume_db,
                    "volume_db": note.volume_db,
                    "volume_limited": note.volume_limited,
                    "voice_end": getattr(note, "voice_end", None),
                }

            def _event_order(note):
                return (note.start, note.chan, note.source_pitch, note.id)

            events = [
                _event_payload(note, converted=True)
                for note in sorted(scheduled_notes, key=_event_order)
            ]
            display_events = [
                _event_payload(note, converted=id(note) in scheduled_ids)
                for note in sorted(notes, key=_event_order)
            ]
            timing = self._current_timing()
            speed = self._doc["tuning"]["playback_speed"]
            # `Song.duration_ms` is authoritative as of Phase 3 -- this used to
            # be recomputed here from grid length and the furthest note every
            # single call, which is exactly the "silently snaps back" behaviour
            # `set_song_length` exists to end. The song's own clock is at speed
            # 1.0, like every other value the window reads off it; dividing by
            # speed is what keeps this number on the same clock as `events`
            # and `timing`, which `source_notes` already scaled the same way.
            duration_ms = int(round(self._song.duration_ms / speed))
            manifest = {
                "duration_ms": duration_ms,
                "source_duration_ms": int(round(timing["source_duration_ms"])),
                "events": events,
                "sounds": sorted({event["sound"] for event in events}),
                "display_events": display_events,
                "release_s": levers["release_s"],
                "hard_stop": levers["hard_stop"],
                "timing": timing,
                # The ruler's brace, on the same clock as `duration_ms` above.
                # Always a real span -- see `Song.loop_start_ms`'s own
                # docstring -- and independent of whether it currently does
                # anything (`loop_enabled` is that switch).
                "loop_start_ms": int(round(self._song.loop_start_ms / speed)),
                "loop_end_ms": int(round(self._song.loop_end_ms / speed)),
                "loop_enabled": bool(self._song.loop_enabled),
            }
            if not with_stats:
                return manifest

            # These are precisely the quantities the status strip and the
            # immediate conversion warnings consume. In particular, their
            # note and voice counts follow the same preparation path as the
            # compiler; only map serialisation (and its byte-size warning) is
            # deferred until the user explicitly dry-runs or exports.
            preview_stats = dict(source_stats)
            preview_stats.update(
                {
                    "notes": len(audible_notes),
                    "decaying": len(prepared.decaying),
                    "sustained": len(prepared.sustained),
                    "voices": prepared.voice_count,
                    "shared_one_shots": len(prepared.shared_decaying),
                    "expressive_notes": len(prepared.sustained) + len(prepared.expressive_decaying),
                    "expressive_one_shots": len(prepared.expressive_decaying),
                    "expressive_voices": prepared.voice_count,
                    "long_sustains": sum(note.duration > 1000 for note in prepared.sustained),
                    "peak_voices": prepared.voice_count,
                    "max_speakers": levers["max_speakers"],
                }
            )
            return manifest, self._report(preview_stats)

    def _report(self, stats) -> dict:
        report = dict(stats)
        report["warnings"] = self._warnings(stats)
        return report

    def export(self) -> dict:
        """Write the map, and say where it went and what was already there.

        `replaced` is read BEFORE the write, which is the only moment it can be
        read at all. `rawmap.json` is a single global slot -- one filename in one
        folder, because that is the only thing the loader reads -- and a button
        invites repeated use in a way a typed command did not. Overwriting
        somebody's other map without saying so is the failure this answers.
        """
        with self._lock:
            raw, stats = self.compile()
            destination = paths.rawmap_destination(self._doc["out_dir"])
            replaced = destination.exists()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            return {
                "destination": str(destination),
                "advice": _advice(destination),
                "replaced": replaced,
                "stats": self._report(stats),
            }

    # ---- the rulers ----

    def _family_for(self, channel):
        """The family a channel will actually be compiled with.

        The chosen one if there is one, the automatic one otherwise -- and the
        automatic one is what makes this worth a method. A ruler or a warning
        that only spoke about families somebody had picked would say nothing at
        all about a file nobody had touched yet, which is every file at the
        moment it opens.
        """
        entry = self._entry_for(channel)
        if entry and entry.get("sound") is not None:
            return None
        chosen = entry.get("family")
        return chosen or channel.auto_family

    def rulers(self) -> dict:
        """Where to draw every channel's notes and its instrument's reach.

        `{channel-as-string: segments}`, and the segments are None for the
        percussion channel, whose lowest and highest are key numbers rather than
        pitches. String keys because this crosses into Javascript, where JSON
        has no integer ones and a reader looking up channel 0 would find nothing
        and draw an empty row without ever raising.
        """
        with self._lock:
            if self._analysis is None:
                return {}
            rulers = {}
            for channel in self._analysis.channels:
                family = self._family_for(channel)
                sample_span = (
                    None if family is None else palette.family_range(family, self._note_index)
                )
                span = (
                    None
                    if sample_span is None
                    else (max(_AXIS[0], sample_span[0] - 24), min(_AXIS[1], sample_span[1] + 24))
                )
                rulers[channel.key] = analysis.ruler_segments(channel, span, _AXIS)
            return rulers

    # ---- warnings ----

    def _parts(self) -> list:
        return list(self._analysis.channels) if self._analysis is not None else []

    def _entry_for(self, info) -> dict:
        """The settings entry governing one part: its own, else its channel's.

        The same precedence the parser applies, restated here because the
        warnings have to describe the arrangement the compile actually
        produces. A bare channel entry is the wildcard covering every part on
        that channel; a part key names one and beats it.
        """
        channels = self._doc["channels"]
        entry = channels.get(info.key)
        if entry is None:
            entry = channels.get(str(info.channel))
        return entry or {}

    def _muted(self) -> set:
        """The keys of parts a mute silences. Keys, not channel numbers: two
        parts can share a channel and only one of them be muted."""
        return {p.key for p in self._parts() if self._entry_for(p).get("muted")}

    def _soloed(self) -> set:
        return {p.key for p in self._parts() if self._entry_for(p).get("soloed")}

    def _inaudible(self) -> set:
        muted = self._muted()
        soloed = self._soloed()
        if not soloed:
            return muted
        return muted | ({p.key for p in self._parts()} - soloed)

    def _who(self, channel: int) -> str:
        """A channel named the way expression warnings identify a track."""
        for info in self._analysis.channels if self._analysis is not None else ():
            if info.channel == channel:
                return "Channel %d (%s)" % (channel, info.program_name)
        return "Channel %d" % channel

    def _unmapped_drum_keys(self) -> list:
        """Percussion keys the file plays that nothing has a sound for.

        `DRUM_MAP` drops the exotic keys rather than guessing at them, so this is
        the ordinary way a file loses notes. Keys the user has already given a
        sound are excluded, or the warning would go on naming a row that has
        been dealt with.
        """
        given = {int(key) for key in self._doc["drum_keys"]}
        keys = set()
        inaudible = self._inaudible()
        for channel in self._analysis.channels:
            if channel.key in inaudible:
                continue
            keys.update(
                key
                for key, shader in channel.drum_keys.items()
                if shader is None and key not in given
            )
        return sorted(keys)

    @staticmethod
    def _note_name(note: int) -> str:
        names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        note = int(note)
        return "%s%d" % (names[note % 12], note // 12 - 1)

    def _multi_source_warnings(self) -> list[str]:
        """Name any track whose exact sound is several recordings in a trench coat.

        One DOOM event name can be backed by several distinct recordings, and
        the engine plays a DIFFERENT one per trigger. Nothing in a map selects
        or compensates for that choice, and each recording carries its own
        inherent pitch, so a calibrated root cannot hold: the same correct
        semitone modifier lands at a different audible pitch on most notes.

        This has to be said here because it is invisible everywhere else. The
        window can only extract and audition the FIRST recording, so preview
        sounds consistent and correct no matter how wrong the export will be --
        which is exactly how it cost a full debugging session before the live
        probe settled it. Proven live; see doom-re
        `docs/truth/engine/snapmap-timeline-sound-modifiers.md`.
        """
        try:
            from snapmap_midi.audio import library
        except Exception:
            return []
        warnings = []
        for info in self._parts():
            entry = self._entry_for(info)
            sound = entry.get("sound")
            if not sound:
                continue
            try:
                count = library.event_source_count(sound)
            except Exception:
                continue
            if count <= 1:
                continue
            # Only pitch-following tracks are actually harmed. A track playing
            # the sound unpitched gets variation, which is usually the point of
            # a multi-recording event and not worth a warning.
            if entry.get("pitch_follow"):
                warnings.append(
                    "%s plays %s, which is %d different recordings under one name. The game "
                    "picks a different one per note and no map setting can choose or correct "
                    "for it, so this track will play at randomized pitches in game however "
                    "right it sounds here -- only the first recording is auditioned. Use a "
                    "single-recording sound for a pitched part."
                    % (self._who(info.channel), sound, count)
                )
            else:
                warnings.append(
                    "%s plays %s, which is %d different recordings under one name. The game "
                    "picks a different one per note, so it will vary in game while preview "
                    "always plays the first. Harmless unpitched; do not calibrate a pitch "
                    "against it." % (self._who(info.channel), sound, count)
                )
        return warnings

    def _octave_fold_notices(self, stats) -> list[str]:
        """Say which tracks were moved, and by how much.

        A fold rescues the melody but moves the part, so it cannot be silent.
        The alternative was the bug it replaces: every note past the engine's
        limit clamping to the same modifier, a whole line flattening onto one
        pitch with nothing anywhere saying why.
        """

        notices = []
        for detail in stats.get("octave_shift_channels") or []:
            octaves = detail["octaves"]
            notices.append(
                "%s sits too far from its sound's natural note for SnapMap's -24 to +24 "
                "semitone range, so it plays %d octave%s %s written. The tuning is exact -- "
                "an octave is a whole 12 semitones, so calibration cents are unchanged. Set "
                "the track's Playback octave to pick a different one."
                % (
                    self._who(detail["channel"]),
                    abs(octaves),
                    "" if abs(octaves) == 1 else "s",
                    "above" if octaves > 0 else "below",
                )
            )
        return notices

    def _pitch_limit_warnings(self, stats) -> list[str]:
        details = stats.get("pitch_limit_channels") or []
        if not details:
            return [
                "%d notes need more than SnapMap's -24 to +24 semitone range. "
                "Their pitch is clamped at the nearest engine limit." % stats["pitch_limited"]
            ]
        warnings = []
        for detail in details:
            low = self._note_name(detail["source_low"])
            high = self._note_name(detail["source_high"])
            notes = low if low == high else "%s-%s" % (low, high)
            requested = (
                str(detail["requested_low"])
                if detail["requested_low"] == detail["requested_high"]
                else "%+g to %+g" % (detail["requested_low"], detail["requested_high"])
            )
            warnings.append(
                "%s: %d note%s (%s) request%s %s semitones, outside SnapMap's -24 to +24 "
                "semitone range, so playback and export clamp at the nearest limit. Change the "
                "channel pitch reference or the affected note offsets."
                % (
                    self._who(detail["channel"]),
                    detail["count"],
                    "" if detail["count"] == 1 else "s",
                    notes,
                    "s" if detail["count"] == 1 else "",
                    requested,
                )
            )
        return warnings

    def _warnings(self, stats) -> list:
        """Plain-language problems, each with the number that decides whether to
        care and the lever that changes it.

        The thresholds are `docs/limits.md`'s, not invented here. Neutral
        decaying notes hold no dedicated emitter, while expressive one-shots
        reserve one for their measured or fallback tail and are included in the
        same song-wide peak as sustains. Total event count still says nothing
        about simultaneous pressure: a long sequence can be cheap and one dense
        chord expensive. The quantities below describe duration and peak voice
        use instead.

        Ordered by what each costs a listener. Silence first, then notes that do
        not play, then notes that play at the wrong pitch, then the smearing the
        emitter limit produces, then passages the compiler thinned on purpose.
        """
        channels = self._analysis.channels if self._analysis is not None else []
        muted = self._muted()
        inaudible = self._inaudible()
        warnings = []

        if channels and all(channel.key in muted for channel in channels):
            warnings.append("Nothing will play: all %d channels are muted." % len(channels))
        elif channels and all(channel.key in inaudible for channel in channels):
            warnings.append(
                "Nothing will play: mute and solo settings exclude all %d channels." % len(channels)
            )
        elif not stats["notes"]:
            warnings.append("Nothing will play: no note in this file has a sound in the palette.")
        if stats["dropped"]:
            keys = self._unmapped_drum_keys() if channels else []
            text = "%d notes have no sound and will not play." % stats["dropped"]
            if keys:
                # Named only when there are keys to name. Percussion is very
                # nearly the only way a note reaches this count -- every family
                # the picker offers has a sound for some pitch -- but claiming
                # unmapped percussion keys where there are none would send the
                # reader toward a track that does not need changing.
                text += (
                    " Percussion keys %s are unmapped -- assign that channel an instrument "
                    "set or exact sound, or add per-key drum_keys in the sidecar."
                    % ", ".join(str(key) for key in keys)
                )
            warnings.append(text)

        warnings.extend(self._multi_source_warnings())

        warnings.extend(self._octave_fold_notices(stats))

        if stats.get("pitch_limited"):
            warnings.extend(self._pitch_limit_warnings(stats))
        if stats.get("volume_limited"):
            warnings.append(
                "%d notes exceed SnapMap's -60 to +20 dB output range after note and global "
                "volume. Their loudness is clamped." % stats["volume_limited"]
            )

        if stats["long_sustains"]:
            # Counted from the sustained notes before `max_poly` thinning, so
            # with that lever set this is an upper bound rather than a tally.
            # Erring high is the right side for a warning about a risk, and the
            # sentence claims no more than the number can carry.
            warnings.append(
                "%d sustained notes hold longer than a second. Past about a second the engine "
                "may recycle the emitter, and a recycled note rings to the end of its sample "
                "under the next phrase. Cap the sustain, or move the family to the decaying "
                "path." % stats["long_sustains"]
            )

        if stats["peak_voices"] >= stats["max_speakers"]:
            warnings.append(
                "The song used all %d global voices. Its densest passages may cut ringing "
                "notes short; raise Global Voices, or cap Track Voices or track Polyphony."
                % stats["max_speakers"]
            )

        # Last, because it is the only one here that says nothing about how the
        # song SOUNDS. The map loads and the music plays exactly as written --
        # what is lost is the ability to open the timeline in the editor, and
        # the editor's own message for that is "could not open this timeline"
        # with no size in it and no hint that size is the reason.
        budget = stats.get("timeline_budget")
        size = stats.get("timeline_bytes")
        if budget and size and size > budget:
            if stats.get("timeline_sharding_enabled"):
                warnings.append(
                    "The largest timeline shard is %s and will play, but SnapMap cannot open it "
                    "for editing: the editor serializes one entity into a fixed buffer of about "
                    "%s and gives up past it. Shorten the densest simultaneous passage, mute a "
                    "track, or cap its polyphony to bring this indivisible batch under the limit."
                    % (_kb(size), _kb(budget))
                )
            else:
                warnings.append(
                    "The single song timeline is %s and will play, but SnapMap cannot open it "
                    "for editing: the editor serializes one entity into a fixed buffer of about "
                    "%s and gives up past it. Automatic timeline sharding is currently disabled; "
                    "shorten the arrangement or reduce exported events if editor access is "
                    "required." % (_kb(size), _kb(budget))
                )
        return warnings
