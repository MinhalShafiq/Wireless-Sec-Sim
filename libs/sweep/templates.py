"""Apply a sweep template to a base Scenario."""

from __future__ import annotations

from libs.schemas import Scenario
from libs.sim_engine.adversaries import add_jammer, add_rogue_gnb


def apply_template(
    base: Scenario, template: str, params: dict, *, seed: int, run_id: str
) -> Scenario:
    """Return a new Scenario with ``params`` applied per ``template``.

    Always sets ``propagation.seed`` and ``name`` based on ``seed`` /
    ``run_id``. The base is never mutated.
    """
    s = base.model_copy(deep=True)
    s.propagation.seed = seed
    s.name = run_id

    if template == "none":
        return s

    pos = (
        float(params.get("position_x", 0.0)),
        float(params.get("position_y", 0.0)),
        float(params.get("position_z", 8.0)),
    )

    if template == "jammer":
        return add_jammer(
            s, pos,
            power_dbm=float(params.get("power_dbm", 50.0)),
            name="jammer0",
        )

    if template == "rogue":
        # Optional power override; defaults to mimic of base's first legitimate TX.
        kwargs = {"name": "rogue0"}
        out = add_rogue_gnb(s, pos, **kwargs)
        if "power_dbm" in params:
            out.transmitters[-1].power_dbm = float(params["power_dbm"])
        return out

    if template == "legit_tx":
        # Mutate one of the base's legitimate TXs in place — no adversary added.
        # Useful for generating diverse benign data: same scene, different deployment.
        target_idx = int(params.get("target_idx", 0))
        tx = s.transmitters[target_idx]
        if tx.role != "legitimate":
            raise ValueError(
                f"legit_tx template targets index {target_idx} which has role={tx.role}"
            )
        if "position_x" in params:
            tx.position.x = float(params["position_x"])
        if "position_y" in params:
            tx.position.y = float(params["position_y"])
        if "position_z" in params:
            tx.position.z = float(params["position_z"])
        if "power_dbm" in params:
            tx.power_dbm = float(params["power_dbm"])
        return s

    raise ValueError(f"unknown sweep template: {template!r}")
