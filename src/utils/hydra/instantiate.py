from typing import Any

from hydra._internal.utils import _get_cls_name, _locate
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


def instantiate_recursive(  # noqa: max-complexity=13
    cfg: DictConfig | ListConfig,
) -> Any:
    """Instantiate objects which can have arbitrary
    nested _target_ parameters in the config.

    We implemented it only because  we cannot update hydra
    to the newer version because of fairseq hardcoded dependency

    Args:
        cfg (DictConfig): config of the object to instantiate

    Returns:
        Any: instantiated object of type _target_ from the config root
    """

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items():
            if isinstance(v, DictConfig):
                if "_target_" in v:
                    cfg = OmegaConf.to_container(cfg, resolve=True)
                    cfg[k] = instantiate_recursive(v)
            elif isinstance(v, ListConfig):
                inst_list = instantiate_recursive(v)
                cfg = OmegaConf.to_container(cfg, resolve=True)
                cfg[k] = inst_list

        if "_args_" in cfg:
            args = cfg.pop("_args_")
            cls = _get_cls_name(cfg)
            kwargs = cfg
            type_or_callable = _locate(cls)
            return type_or_callable(*args, **kwargs)
        elif isinstance(cfg, dict):
            cls = _get_cls_name(cfg)
            kwargs = cfg
            type_or_callable = _locate(cls)
            return type_or_callable(**kwargs)
        else:
            return instantiate(cfg)

    elif isinstance(cfg, ListConfig):
        inst_cfg = []
        for v in cfg:
            if isinstance(v, (DictConfig, ListConfig)):
                inst_cfg.append(instantiate_recursive(v))
            else:
                inst_cfg.append(v)
        return inst_cfg
