"""
Audio Feedback Engine for real-time auditory exercise coaching.
Runs sound synthesis on a background worker thread to ensure zero video lag.
"""

from __future__ import annotations
import threading
import sys
import time
from typing import Optional


class AudioFeedback:
    """Asynchronous sound cue generator for rep completions and posture warnings."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._is_windows = sys.platform.startswith("win")
        self._last_alert_time = 0.0

    def toggle(self) -> bool:
        """Toggle audio state on/off."""
        self.enabled = not self.enabled
        return self.enabled

    def _play_async(self, func, *args):
        if not self.enabled:
            return
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()

    def play_rep_completed(self, is_good: bool = True):
        """Play sound when a repetition is counted."""
        def _beep():
            try:
                if self._is_windows:
                    import winsound
                    if is_good:
                        winsound.Beep(1050, 110)
                    else:
                        winsound.Beep(450, 220)
                else:
                    # ANSI bell fallback on Unix/Mac
                    sys.stdout.write('\a')
                    sys.stdout.flush()
            except Exception:
                pass

        self._play_async(_beep)

    def play_posture_alert(self, cooldown: float = 1.5):
        """Play cautionary tone for bad posture with rate limiting cooldown."""
        now = time.time()
        if now - self._last_alert_time < cooldown:
            return
        self._last_alert_time = now

        def _alert():
            try:
                if self._is_windows:
                    import winsound
                    winsound.Beep(380, 160)
            except Exception:
                pass

        self._play_async(_alert)

    def play_milestone(self):
        """Play celebration chime when hitting milestones."""
        def _milestone():
            try:
                if self._is_windows:
                    import winsound
                    winsound.Beep(880, 90)
                    time.sleep(0.04)
                    winsound.Beep(1320, 140)
            except Exception:
                pass

        self._play_async(_milestone)
