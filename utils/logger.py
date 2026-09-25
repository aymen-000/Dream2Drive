import time
import collections


class Logger:
    def __init__(self, log_every: int = 20):
        self.log_every = log_every
        self._history = collections.defaultdict(list)
        self._t0 = time.time()

    def log(self, step: int, **scalars):
        for k, v in scalars.items():
            self._history[k].append(float(v))
        if step % self.log_every == 0:
            elapsed = time.time() - self._t0
            msg = f"[step {step:>7d} | {elapsed:8.1f}s] "
            parts = []
            for k, vals in self._history.items():
                recent = vals[-self.log_every:]
                parts.append(f"{k}={sum(recent) / len(recent):.4f}")
            print(msg + "  ".join(parts))

    def history(self):
        return dict(self._history)
