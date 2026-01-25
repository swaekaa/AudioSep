# F:\AudioSep\bandit\sub_test.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

import typing
from pprint import pprint
import copy
import torch
import pytorch_lightning as pl

from core import LightningSystem, data
from utils.config import dict_to_trainer_kwargs, read_nested_yaml

# Tiny perf boost on RTX Tensor Cores
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _infer_stems_from_cfg(system_cfg: dict) -> list[str]:
    try:
        stems = list(system_cfg.get("model", {}).get("kwargs", {}).get("band_specs_map", {}).keys())
        if stems:
            return stems
    except Exception:
        pass
    return ["speech", "music", "effects"]


def _to_bct(x: torch.Tensor) -> torch.Tensor:
    """
    Coerce to shape [B, C, T]:
      - [B, S, C, T] isn't handled here (that split happens earlier);
      - [B, C, T]  -> as-is,
      - [C, T]     -> add batch dim,
      - [T]        -> add batch+channel dims.
    """
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unexpected tensor shape for BCT coercion: {tuple(x.shape)}")


class _IdentityFader:
    """
    No-op fader that *normalizes* model output to:
        {"audio": {stem: tensor[B, C, T], ...}}
    and ensures all expected stems exist by zero-filling missing ones.
    """
    def __init__(self, stems: list[str]):
        self.stems = stems or ["speech", "music", "effects"]

    def __call__(self, y, *args, **kwargs):
        out = self.normalize(y)
        out["audio"] = self._ensure_all_stems(out.get("audio", {}))
        return out

    def normalize(self, y):
        # Already dict with "audio"
        if isinstance(y, dict):
            if "audio" in y:
                audio_val = y["audio"]
                if isinstance(audio_val, dict):
                    # Coerce each tensor to [B, C, T]
                    coerced = {}
                    for k, v in audio_val.items():
                        if torch.is_tensor(v):
                            coerced[k] = _to_bct(v)
                    return {"audio": coerced}
                if torch.is_tensor(audio_val):
                    return {"audio": self._tensor_to_stem_dict(audio_val)}
                # Unknown structure -> empty
                return {"audio": {}}

            # Dict of stems -> tensors
            if all(isinstance(k, str) for k in y.keys()):
                coerced = {}
                for k, v in y.items():
                    if torch.is_tensor(v):
                        coerced[k] = _to_bct(v)
                return {"audio": coerced}

            # Unknown dict
            return {"audio": {}}

        # Tensor or list/tuple paths
        if torch.is_tensor(y):
            return {"audio": self._tensor_to_stem_dict(y)}

        if isinstance(y, (list, tuple)) and len(y) > 0 and torch.is_tensor(y[0]):
            d = {}
            for i, t in enumerate(y):
                key = self.stems[i] if i < len(self.stems) else f"stem_{i}"
                d[key] = _to_bct(t)
            return {"audio": d}

        # Fallback
        return {"audio": {}}

    def _tensor_to_stem_dict(self, t: torch.Tensor) -> dict:
        # [B, S, C, T] -> split across S
        if t.dim() == 4:
            B, S, C, T = t.shape
            d = {}
            for i in range(S):
                key = self.stems[i] if i < len(self.stems) else f"stem_{i}"
                d[key] = t[:, i, :, :]
            return d
        # [B, C, T] or lower -> single stem
        key = self.stems[0] if self.stems else "stem_0"
        return {key: _to_bct(t)}

    def _ensure_all_stems(self, d: dict) -> dict:
        """
        Ensure every expected stem exists; if missing, create zeros
        matching the shape of an available stem (B, C, T).
        """
        d = dict(d) if isinstance(d, dict) else {}
        # Find a reference shape
        ref = None
        for v in d.values():
            if torch.is_tensor(v):
                ref = _to_bct(v)
                break
        # If nothing is present, make a minimal reference
        if ref is None:
            ref = torch.zeros(1, 1, 1, dtype=torch.float32)

        for stem in self.stems:
            if stem not in d or not torch.is_tensor(d[stem]):
                d[stem] = torch.zeros_like(ref)

            else:
                d[stem] = _to_bct(d[stem])
        return d


def test(
    config_path: str,
    ckpt_path: typing.Optional[str] = None,
    **kwargs: typing.Any,
) -> None:
    # --- Load & echo config
    cfg = read_nested_yaml(config_path)
    pprint(copy.deepcopy(cfg))

    # --- Build datamodule
    data_cfg = copy.deepcopy(cfg["data"]["data"])
    dm_name = data_cfg.pop("datamodule")
    datamodule = data.__dict__[dm_name](**data_cfg)

    # --- Trainer kwargs from YAML; drop logger/callbacks (we override them)
    trainer_kwargs = dict_to_trainer_kwargs(cfg["trainer"])
    trainer_kwargs.pop("logger", None)
    trainer_kwargs.pop("callbacks", None)

    trainer = pl.Trainer(
        **trainer_kwargs,
        logger=False,                 # no external logging for this test run
        enable_model_summary=False,
        enable_progress_bar=True,
        callbacks=[],                 # no YAML callbacks
        **kwargs,
    )

    # --- Build/Load model
    system_cfg = cfg["system"]
    if ckpt_path:
        model = LightningSystem.load_from_checkpoint(
            ckpt_path,
            config=system_cfg,
            loss_adjustment=1.0,
            strict=False,
            map_location="cpu",        # safer: load on CPU; PL will place on device
        )
    else:
        model = LightningSystem(system_cfg, loss_adjustment=1.0)

    # --- Stems list (used by IdentityFader)
    stems = _infer_stems_from_cfg(system_cfg)

    # --- Disable fader but keep a *normalizing* identity in place
    if not hasattr(model, "fader_config") or not isinstance(model.fader_config, dict):
        model.fader_config = {}
    model.fader_config.update({"enabled": False, "name": None})
    model.fader = _IdentityFader(stems=stems)

    # Monkeypatch attach_fader: only attach real fader if enabled+named; else keep identity
    orig_attach_fader = getattr(model, "attach_fader", None)

    def _attach_fader_safe(*args, **kws):
        cfg_local = getattr(model, "fader_config", {}) or {}
        enabled = isinstance(cfg_local, dict) and cfg_local.get("enabled") and cfg_local.get("name")
        if not enabled:
            print("[info] Fader disabled or no name provided — using identity fader.")
            model.fader = _IdentityFader(stems=stems)
            return
        if orig_attach_fader:
            return orig_attach_fader(*args, **kws)

    model.attach_fader = _attach_fader_safe

    # If YAML explicitly enables a named fader, attach it; else identity remains
    fader_cfg = system_cfg.get("inference", {}).get("fader", {})
    if isinstance(fader_cfg, dict) and fader_cfg.get("enabled") and fader_cfg.get("name"):
        model.attach_fader()

    # --- Run test
    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    import fire
    fire.Fire(test)
