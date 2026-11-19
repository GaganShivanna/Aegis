"""Where checkpoint data lives.

Three tiers, each a set of locations:

    local    nodes/<node>/local/step_N/    primary copy, on the node that ran the rank
    peer     nodes/<node>/peer/step_N/     replica of a neighbouring node's shard
    durable  durable/step_N/               object storage, shared by everyone

A node's local and peer directories die with the node. Durable storage
survives node loss but is the slowest to read. That asymmetry is what the
recovery planner trades off.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from aegis.checkpoint.store import step_dir_name, step_from_dir

LOST_MARKER = ".lost-"


class Tier(StrEnum):
    LOCAL = "local"
    PEER = "peer"
    DURABLE = "durable"


@dataclass(frozen=True, order=True, slots=True)
class Location:
    tier: Tier
    node: str = ""

    def __str__(self) -> str:
        return str(self.tier) if self.tier is Tier.DURABLE else f"{self.tier}@{self.node}"

    @classmethod
    def parse(cls, raw: str) -> Location:
        if "@" in raw:
            tier, node = raw.split("@", 1)
            return cls(Tier(tier), node)
        return cls(Tier(raw))

    @property
    def node_scoped(self) -> bool:
        return self.tier is not Tier.DURABLE


class Layout:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    @property
    def nodes_dir(self) -> Path:
        return self.root / "nodes"

    @property
    def durable_dir(self) -> Path:
        return self.root / "durable"

    @property
    def tier_state_file(self) -> Path:
        return self.root / "tier_state.json"

    def tier_root(self, location: Location) -> Path:
        if location.tier is Tier.DURABLE:
            return self.durable_dir
        return self.nodes_dir / location.node / str(location.tier)

    def dir_for(self, location: Location, step: int) -> Path:
        return self.tier_root(location) / step_dir_name(step)

    def known_nodes(self) -> list[str]:
        if not self.nodes_dir.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.nodes_dir.iterdir()
            if entry.is_dir() and LOST_MARKER not in entry.name
        )

    def locations(self, *, exclude_nodes: frozenset[str] | set[str] = frozenset()) -> list[Location]:
        found = [
            Location(tier, node)
            for node in self.known_nodes()
            if node not in exclude_nodes
            for tier in (Tier.LOCAL, Tier.PEER)
        ]
        found.append(Location(Tier.DURABLE))
        return found

    def steps_in(self, location: Location) -> list[int]:
        root = self.tier_root(location)
        if not root.is_dir():
            return []
        steps = [
            step
            for entry in root.iterdir()
            if entry.is_dir() and (step := step_from_dir(entry.name)) is not None
        ]
        return sorted(steps, reverse=True)

    def lose_node(self, node: str) -> Path | None:
        """Fault injection: make a node's disks disappear.

        Renamed rather than deleted so a post-mortem can still see what was
        there.
        """
        source = self.nodes_dir / node
        if not source.exists():
            return None
        target = self.nodes_dir / f"{node}{LOST_MARKER}{time.time_ns()}"
        os.replace(source, target)
        return target


def peer_of(rank: int, placement: list[str] | tuple[str, ...]) -> str | None:
    """Ring replication: rank r's shard is copied to the node running r+1.

    Any single node loss then leaves at least one copy of every shard on the
    survivors, without needing a spare node to hold replicas.
    """
    if len(placement) < 2:
        return None
    peer = placement[(rank + 1) % len(placement)]
    return None if peer == placement[rank] else peer
