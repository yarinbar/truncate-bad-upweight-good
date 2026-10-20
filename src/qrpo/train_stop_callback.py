from __future__ import annotations

from transformers.trainer_callback import TrainerCallback


class StopAfterStepsCallback(TrainerCallback):
    def __init__(self, stop_after_steps: int):
        self.stop_after_steps = int(stop_after_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after_steps > 0 and state.global_step >= self.stop_after_steps:
            control.should_save = True
            control.should_training_stop = True
        return control
