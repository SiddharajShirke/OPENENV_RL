# rl/

Reinforcement learning module.

- `feature_builder.py`: ObservationModel to flat vector
- `action_mask.py`: structural action validity masks
- `gym_wrapper.py`: Gymnasium adapter around GovWorkflowEnv
- `curriculum.py`: staged task scheduler (Phase 2)
- `train_ppo.py`: Phase 1 and Phase 2 training entrypoint
- `train_recurrent.py`: Phase 3 recurrent PPO entrypoint
- `evaluate.py`: model evaluation on grader metrics
- `callbacks.py`: eval and cost-monitor callbacks
- `configs/`: YAML configs for PPO training
