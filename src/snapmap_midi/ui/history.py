"""Undo and redo: a linear stack of things done and how to take them back.

Session-only, and deliberately so. Closing a project and reopening it starts a
clean history, the way every editor a musician has ever used behaves -- the
alternative is a saved file carrying an unbounded record of every mistake
somebody made on the way to it.

Linear rather than a tree. A redo branch is discarded the moment a new command
lands on top of it, which is the behaviour anybody reaching for Ctrl+Z expects
and the only one that can be described in one sentence.

Nothing here knows what a song is. A command is a pair of callables and a label,
so the mutations Phase 2 adds -- move, resize, delete, draw, add a track -- push
onto this without it needing to learn what any of them mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

#: How many steps back a session keeps. Deep enough that reaching the end is a
#: surprise, shallow enough that a long editing session cannot pin an entire
#: song's worth of superseded state in memory.
DEFAULT_LIMIT = 200


@dataclass
class Command:
    """One reversible change: what it was called, how to do it, how to undo it.

    `apply` is called by `redo` and, if the caller asks, by `do`. It must be
    safe to call more than once, because a redo is exactly that.
    """

    label: str
    apply: Callable[[], None]
    revert: Callable[[], None]


class History:
    """A linear undo stack with a cursor.

    Everything before the cursor has been applied; everything from the cursor on
    has been undone and is waiting to be redone. There is no separate redo list,
    because two lists is two things that can disagree about what the document
    currently is.
    """

    def __init__(self, limit: int = DEFAULT_LIMIT):
        self._limit = limit
        self._commands: list = []
        self._cursor = 0

    def clear(self) -> None:
        """Forget everything. Opening a different song is a different history."""
        self._commands = []
        self._cursor = 0

    def push(self, command: Command) -> None:
        """Record a change the caller has ALREADY made.

        Push-after-apply rather than apply-on-push, because a mutation usually
        has to run first to know what its inverse is -- deleting a note cannot
        describe how to put it back until it has seen the note.
        """
        # Anything undone is now unreachable: the timeline just forked, and this
        # is the branch the user is on.
        del self._commands[self._cursor :]
        self._commands.append(command)
        if len(self._commands) > self._limit:
            # Drop from the far end, which is the oldest thing anybody could
            # still want back.
            del self._commands[: len(self._commands) - self._limit]
        self._cursor = len(self._commands)

    def do(self, command: Command) -> None:
        """Apply a change and record it, for a command that can describe itself
        before it has run."""
        command.apply()
        self.push(command)

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._commands)

    @property
    def undo_label(self):
        return self._commands[self._cursor - 1].label if self.can_undo else None

    @property
    def redo_label(self):
        return self._commands[self._cursor].label if self.can_redo else None

    def undo(self):
        """Take back the last change. Returns its label, or None if there was none."""
        if not self.can_undo:
            return None
        self._cursor -= 1
        command = self._commands[self._cursor]
        command.revert()
        return command.label

    def redo(self):
        """Put back the last undone change. Returns its label, or None."""
        if not self.can_redo:
            return None
        command = self._commands[self._cursor]
        command.apply()
        self._cursor += 1
        return command.label

    def state(self) -> dict:
        """What the Edit menu needs to draw itself."""
        return {
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "undo_label": self.undo_label,
            "redo_label": self.redo_label,
        }
