from __future__ import annotations

import threading
import uuid


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._callbacks = {}

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self):
        if self._event.is_set():
            return
        self._event.set()
        with self._lock:
            callbacks = list(self._callbacks.values())
        for callback in callbacks:
            try:
                callback()
            except Exception:
                continue

    def register(self, callback):
        callback_id = str(uuid.uuid4())
        with self._lock:
            if self.cancelled:
                callback()
            else:
                self._callbacks[callback_id] = callback

        def unregister():
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return unregister

    def wait(self, seconds: float) -> bool:
        return self._event.wait(seconds)
