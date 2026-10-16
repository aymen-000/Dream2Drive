# Dream2Drive

**Learning Temporally Abstract World Models for Zero-Shot F1TENTH Racing**

Dream2Drive adapts OPOSM to F1TENTH autonomous racing. It learns latent driving skills, a low-level policy, and a skill-conditioned temporally abstract world model from offline driving logs. At evaluation, a planner uses imagined LiDAR rollouts to choose skills without online training.

## Method

An encoder maps each offline trajectory segment to a latent skill. The skill-conditioned policy decodes that skill into low-level steering and speed commands, while the world model predicts the terminal LiDAR scan after executing it. The model is trained offline with a causal EM objective. At run time, `planning.mode` selects either:

- `cem`: Cross-Entropy Method search over sequences of latent skills.
- `direct`: execute the conditional prior-mean skill once per observation, without CEM search.

## Results

One run was evaluated on each of the 19 maps listed below. Success means completing one lap; collision rate is the fraction of runs with a collision.

| Planning mode | Lap completion | Collisions | Maps |
|---|---:|---:|---:|
| Direct (prior mean) | **17/19 (89.5%)** | **2/19 (10.5%)** | 19 |
| CEM (MPC) | 11/19 (57.9%) | 8/19 (42.1%) | 19 |

The direct-mode failures were on Montreal and Shanghai. CEM-mode failures were on Budapest, Montreal, Monza, SaoPaulo, Shanghai, Silverstone, Spa, and YasMarina. Evaluation maps: BrandsHatch, Budapest, Catalunya, Hockenheim, IMS, Melbourne, Montreal, Monza, Nuerburgring, Oschersleben, Sakhir, SaoPaulo, Sepang, Shanghai, Silverstone, Sochi, Spa, YasMarina, and Zandvoort.

## Reproduction

Install dependencies and obtain the F1TENTH simulator and track maps, then configure paths and hyperparameters in [`configs/default.yaml`](configs/default.yaml).

```bash
pip install -r requirements.txt
python scripts/generate_offline_data.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml \
  --em_checkpoint checkpoints/oposm_final.pt \
  --normalizer_dir checkpoints \
  --eval_maps BrandsHatch Budapest Catalunya Hockenheim IMS Melbourne Montreal Monza Nuerburgring Oschersleben Sakhir SaoPaulo Sepang Shanghai Silverstone Sochi Spa YasMarina Zandvoort
```

To plan on one map, run `scripts/plan_lap.py` with a trained checkpoint. Set `planning.mode` to `cem` or `direct` in the config.

## PPO comparison baseline

PPO is the project's only online reinforcement-learning baseline. It receives normalized LiDAR ranges and outputs steering and target speed. Its dense reward combines forward centerline progress, cross-track error, a small positive speed term, a per-second time cost, a lap-completion bonus, and a collision penalty. The time cost discourages the policy from standing still. Training uses multiple configured collection maps and randomized centerline start positions. Evaluation uses seeded randomized start positions by default, so repeated episodes cover different parts of each map and all checkpoints see the same evaluation states. Use `--fixed_start` for a fixed start pose. The map lists, environment settings, and PPO hyperparameters should be reported with any results.

Install the dependencies, then train one model per seed. `--timesteps` is the total number of transitions across vector environments.

```bash
python scripts/train_rl.py --config configs/default.yaml --timesteps 2000000 --seed 0
python scripts/train_rl.py --config configs/default.yaml --timesteps 2000000 --seed 1
python scripts/train_rl.py --config configs/default.yaml --timesteps 2000000 --seed 2
```

Each run writes a checkpoint and adjacent JSON metadata under `checkpoints/rl/`. Evaluate the independent seeds on the same held-out maps and episode seeds, optionally alongside previously saved planner evaluations:

```bash
python scripts/evaluate_rl.py --config configs/default.yaml \
  --ppo_models checkpoints/rl/ppo_f1tenth_seed0.zip \
               checkpoints/rl/ppo_f1tenth_seed1.zip \
               checkpoints/rl/ppo_f1tenth_seed2.zip \
  --episodes_per_map 5 --seed 10000 \
  --direct_results results/planner_reference.json \
  --cem_results results/planner_reference.json
```


## Acknowledgements and license

Built on [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) and [F1TENTH racetracks](https://github.com/f1tenth/f1tenth_racetracks). Licensed under MIT.
