import sys
import types
import unittest
from pathlib import Path

try:
    from transformers.trainer_callback import TrainerControl, TrainerState
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    trainer_callback_stub = types.ModuleType("transformers.trainer_callback")

    class TrainerControl:
        def __init__(self):
            self.should_save = False
            self.should_training_stop = False

    class TrainerState:
        def __init__(self):
            self.global_step = 0

    class TrainerCallback:
        pass

    trainer_callback_stub.TrainerCallback = TrainerCallback
    trainer_callback_stub.TrainerControl = TrainerControl
    trainer_callback_stub.TrainerState = TrainerState
    transformers_stub.trainer_callback = trainer_callback_stub
    sys.modules.setdefault("transformers", transformers_stub)
    sys.modules.setdefault("transformers.trainer_callback", trainer_callback_stub)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers.trainer_callback import TrainerControl, TrainerState

from qrpo.train_stop_callback import StopAfterStepsCallback


class StopAfterStepsCallbackTests(unittest.TestCase):
    def test_does_not_stop_before_cutoff(self):
        callback = StopAfterStepsCallback(stop_after_steps=200)
        state = TrainerState()
        state.global_step = 199
        control = TrainerControl()

        updated = callback.on_step_end(args=None, state=state, control=control)

        self.assertFalse(updated.should_save)
        self.assertFalse(updated.should_training_stop)

    def test_requests_save_and_stop_at_cutoff(self):
        callback = StopAfterStepsCallback(stop_after_steps=200)
        state = TrainerState()
        state.global_step = 200
        control = TrainerControl()

        updated = callback.on_step_end(args=None, state=state, control=control)

        self.assertTrue(updated.should_save)
        self.assertTrue(updated.should_training_stop)


if __name__ == "__main__":
    unittest.main()
