"""Helpers to inject adversaries into a Scenario without mutating it.

Each helper returns a deep-copied Scenario with the new actor appended,
so a single baseline can be reused to spawn jammed / rogue / eavesdropped
variants for security experiments.
"""

from __future__ import annotations

from libs.schemas import ReceiverCfg, Scenario, TransmitterCfg, Vec3


def add_jammer(
    scenario: Scenario,
    position: Vec3 | tuple[float, float, float],
    *,
    power_dbm: float = 50.0,
    name: str = "jammer0",
) -> Scenario:
    """Append a jammer TX (omnidirectional, high power)."""
    pos = position if isinstance(position, Vec3) else Vec3(x=position[0], y=position[1], z=position[2])
    new = scenario.model_copy(deep=True)
    new.transmitters.append(
        TransmitterCfg(name=name, position=pos, power_dbm=power_dbm, role="jammer")
    )
    return new


def add_rogue_gnb(
    scenario: Scenario,
    position: Vec3 | tuple[float, float, float],
    *,
    name: str = "rogue0",
    mimic: str | None = None,
) -> Scenario:
    """Append a rogue gNB that mimics a legitimate TX's power.

    If ``mimic`` is given, copy that TX's power_dbm; otherwise copy the
    first legitimate TX. Position is the attacker's chosen location.
    """
    pos = position if isinstance(position, Vec3) else Vec3(x=position[0], y=position[1], z=position[2])
    new = scenario.model_copy(deep=True)
    legitimate = [t for t in new.transmitters if t.role == "legitimate"]
    if not legitimate:
        raise ValueError("scenario must contain at least one legitimate TX to mimic")
    target = next((t for t in legitimate if t.name == mimic), legitimate[0])
    new.transmitters.append(
        TransmitterCfg(name=name, position=pos, power_dbm=target.power_dbm, role="rogue")
    )
    return new


def add_eavesdropper(
    scenario: Scenario,
    position: Vec3 | tuple[float, float, float],
    *,
    name: str = "eve0",
) -> Scenario:
    """Append a passive eavesdropper RX."""
    pos = position if isinstance(position, Vec3) else Vec3(x=position[0], y=position[1], z=position[2])
    new = scenario.model_copy(deep=True)
    new.receivers.append(ReceiverCfg(name=name, position=pos, role="eavesdropper"))
    return new
