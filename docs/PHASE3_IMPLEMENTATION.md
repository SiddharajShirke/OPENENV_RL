# Phase 3 Implementation Notes

Phase 3 goal: Recurrent PPO (LSTM policy) to capture temporal dependencies such as SLA trend and escalation history.

## Implemented Components

- `rl/train_recurrent.py`
  - RecurrentPPO training with `MlpLstmPolicy`
  - LSTM hidden size configurable (default 128)
  - curriculum sampling retained (easy -> medium -> hard)
  - optional transfer of compatible policy tensors from best Phase 2 checkpoint
- `rl/configs/recurrent.yaml`
  - declarative recurrent training and curriculum settings
- `rl/evaluate.py`
  - model loading modes: `auto`, `maskable`, `recurrent`
  - recurrent inference path with LSTM state handling + action-mask sanitization
  - helper `compare_recurrent_vs_flat(...)`
- `rl/callbacks.py`
  - `RecurrentEvalCallback` for periodic grader-based checkpointing in Phase 3
  - recurrent best checkpoints saved as `best_grader_recurrent_<task>.zip` (no collision with Phase 2 files)
- `rl/gym_wrapper.py`
  - optional `hard_action_mask` mode (default off) for safe action execution
- `tests/test_rl_evaluate.py`
  - recurrent hidden-state persistence
  - LSTM reset behavior on episode boundary
  - recurrent >= flat comparison utility check

## Commands (using existing .venv313)

- Train Phase 3:
  - `.\\.venv313\\Scripts\\python.exe -m rl.train_recurrent --timesteps 600000 --n-envs 4 --seed 42 --config rl/configs/recurrent.yaml`
- Train Phase 3-v2 (recommended tuning run):
  - `.\\.venv313\\Scripts\\python.exe -m rl.train_recurrent --timesteps 700000 --n-envs 4 --seed 42 --config rl/configs/recurrent_v2.yaml`
- Evaluate Phase 3 model:
  - `.\\.venv313\\Scripts\\python.exe -m rl.evaluate --model results/best_model/phase3_final.zip --episodes 3 --model-type recurrent`
- Evaluate best recurrent checkpoint (saved during Phase 3 eval):
  - `.\\.venv313\\Scripts\\python.exe -m rl.evaluate --model results/best_model/best_grader_recurrent_mixed_urgency_medium.zip --episodes 3 --model-type recurrent`
- Compare recurrent vs flat on medium task:
  - `.\\.venv313\\Scripts\\python.exe -c "from rl.evaluate import compare_recurrent_vs_flat; print(compare_recurrent_vs_flat('results/best_model/phase2_final.zip','results/best_model/phase3_final.zip'))"`
