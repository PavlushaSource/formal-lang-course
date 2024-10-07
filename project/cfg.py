import pyformlang.cfg
from pyformlang.cfg import Epsilon


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    # Determine tmp_eps so that to_normal_form function does not miss it, cause we need weak cnf
    tmp_eps = pyformlang.cfg.Terminal("Epsilon#Terminal")
    while tmp_eps in cfg.terminals or tmp_eps in cfg.variables:
        tmp_eps += "_"

    # terminals and variables auto-detect from productions field
    cfg = pyformlang.cfg.CFG(
        start_symbol=cfg.start_symbol,
        productions={
            pyformlang.cfg.Production(
                head=prod.head,
                body=[
                    cfg_obj if not isinstance(cfg_obj, Epsilon) else tmp_eps
                    for cfg_obj in prod.body
                ],
            )
            for prod in cfg.productions
        },
    )

    cfg = cfg.to_normal_form()

    # replace the epsilons back
    cfg = pyformlang.cfg.CFG(
        start_symbol=cfg.start_symbol,
        productions={
            pyformlang.cfg.Production(
                head=prod.head,
                body=[
                    cfg_obj if cfg_obj != tmp_eps else Epsilon()
                    for cfg_obj in prod.body
                ],
            )
            for prod in cfg.productions
        },
    )
    return cfg
