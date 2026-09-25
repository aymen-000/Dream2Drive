import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from data.dataset import discover_runs, build_run_arrays, train_test_run_split, OfflineSkillDataset
from models.skill_model import OPOSMSkillModel
from training.em_trainer import EMTrainer, EMTrainerConfig
from utils.seed import set_seed
from scripts.config_utils import load_config, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=[None, "causal_em", "naive_vi"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)

    mode = args.mode or cfg.train.mode
    device = resolve_device(cfg.train.device)

    all_runs = discover_runs(cfg.data.root, cfg.data.runs)
    train_names, eval_names = train_test_run_split(all_runs, cfg.data.train_run_frac, seed=cfg.train.seed)
    print(f"[data] {len(all_runs)} runs found (from scripts/generate_offline_data.py). "
          f"train={len(train_names)} eval={len(eval_names)}")

    train_runs, state_norm, action_norm = build_run_arrays(
        root=cfg.data.root, run_names=train_names,
        state_features=cfg.data.state_features, action_features=cfg.data.action_features,
        dropna=cfg.data.dropna, stride=cfg.data.subsample_stride,
    )
    lidar_width = train_runs[0].lidar.shape[1]
    if lidar_width != cfg.data.lidar_dim:
        raise ValueError(
            f"Dataset has {lidar_width} lidar columns but data.lidar_dim is "
            f"{cfg.data.lidar_dim}. Regenerate the offline data with "
            "scripts/generate_offline_data.py; do not train on mixed scan formats."
        )

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    state_norm.save(os.path.join(cfg.train.checkpoint_dir, "state_normalizer.json"))
    action_norm.save(os.path.join(cfg.train.checkpoint_dir, "action_normalizer.json"))
    with open(os.path.join(cfg.train.checkpoint_dir, "train_eval_split.txt"), "w") as f:
        f.write("TRAIN:\n" + "\n".join(train_names) + "\n\nEVAL:\n" + "\n".join(eval_names) + "\n")

    dataset = OfflineSkillDataset(train_runs, window_len=cfg.data.subtraj_len)
    print(f"[data] {len(dataset)} training sub-trajectory windows "
          f"(T={cfg.data.subtraj_len - 1}, stride={cfg.data.subsample_stride}).")

    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                             drop_last=False, num_workers=2, pin_memory=(device == "cuda"))

    model = OPOSMSkillModel(
        state_dim=len(cfg.data.state_features), action_dim=len(cfg.data.action_features),
        z_dim=cfg.model.z_dim, hidden_dim=cfg.model.hidden_dim,
        gru_hidden_dim=cfg.model.gru_hidden_dim, gru_bidirectional=cfg.model.gru_bidirectional,
        min_std=cfg.model.min_std, lidar_dim=cfg.data.lidar_dim,
        obs_embed_dim=cfg.model.obs_embed_dim, lidar_max_range=cfg.data.lidar_max_range,
        tawm_predict_delta=cfg.model.tawm_predict_delta,
    )

    trainer_cfg = EMTrainerConfig(
        lr_e=cfg.train.lr_e, lr_m=cfg.train.lr_m, grad_clip_norm=cfg.train.grad_clip_norm,
        device=device, checkpoint_dir=cfg.train.checkpoint_dir, log_every=cfg.train.log_every,
        save_every_epochs=cfg.train.save_every_epochs, mode=mode,
    )
    trainer = EMTrainer(model, trainer_cfg)

    print(f"[train] mode={mode} device={trainer.device} "
          f"epochs={cfg.train.num_epochs} steps/epoch={cfg.train.steps_per_epoch}")
    trainer.fit(dataloader, num_epochs=cfg.train.num_epochs, steps_per_epoch=cfg.train.steps_per_epoch)
    print("[train] done.")


if __name__ == "__main__":
    main()
