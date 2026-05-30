"""
DAG model and topological operations.

Wraps networkx for in-memory causal graph operations. Neo4j is the persistence
and versioning layer; this module operates on the in-memory mirror used at runtime.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

# Sentinel — distinguishes "no from_val given" from "from_val=None"
_MISSING: Any = object()


class DependencyGraph:
    """
    Directed acyclic graph over dataset attributes.

    Nodes are attribute names. Edges represent causal influence: A -> B means
    "B's value depends on A." The graph must be a DAG — cycles are rejected.

    Hard rules encode deterministic mappings. Two variants:

    1. Unconditional (existing behaviour, backward-compatible)::

        G.add_hard_rule("honorific", when={"sex": "Female"}, value="Ms")

       Fires whenever the parent condition matches, regardless of the node's
       current value.

    2. Conditional transition (new, for sex-typed features)::

        G.add_hard_rule(
            "relationship",
            when={"sex": "Female"},
            value="Wife",
            from_val="Husband",   # only fires when current value IS "Husband"
        )

       The Adult/Census dataset needs this: "Wife" is only the right target
       when the original row had relationship="Husband". Rows with
       relationship="Own-child" or "Not-in-family" should be left alone.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._hard_rules: dict[str, list[dict[str, Any]]] = {}

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_edges(cls, edges: list[tuple[str, str]]) -> "DependencyGraph":
        g = cls()
        for src, dst in edges:
            g.add_edge(src, dst)
        return g

    def add_node(self, name: str, **attrs: Any) -> None:
        self._g.add_node(name, **attrs)

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        self._g.add_edge(src, dst, **attrs)
        if not nx.is_directed_acyclic_graph(self._g):
            self._g.remove_edge(src, dst)
            raise ValueError(f"Adding edge {src} -> {dst} would create a cycle.")

    def add_hard_rule(
        self,
        node: str,
        when: dict[str, Any],
        value: Any,
        from_val: Any = _MISSING,
    ) -> None:
        """
        Register a deterministic rule for *node*.

        Args:
            node:     The descendant node this rule controls.
            when:     Dict of {parent_name: required_value} conditions.
            value:    Target value to assign when the rule fires.
            from_val: Optional. When provided, the rule only fires if the
                      node's CURRENT value (before this propagation step)
                      equals *from_val*. Use for conditional transitions
                      (e.g. Husband -> Wife but not Own-child -> Wife).
        """
        rule: dict[str, Any] = {"when": when, "value": value}
        if from_val is not _MISSING:
            rule["from_val"] = from_val
        self._hard_rules.setdefault(node, []).append(rule)

    # ── topology ────────────────────────────────────────────────────────────

    def parents(self, node: str) -> list[str]:
        return list(self._g.predecessors(node))

    def descendants(self, node: str) -> list[str]:
        return list(nx.descendants(self._g, node))

    def topological_descendants(self, node: str) -> list[str]:
        """Descendants of *node* in topological order (not including *node*)."""
        desc = set(nx.descendants(self._g, node))
        return [n for n in nx.topological_sort(self._g) if n in desc]

    # ── hard rules ───────────────────────────────────────────────────────────

    def has_hard_rule(
        self,
        node: str,
        parent_values: dict[str, Any],
        current_val: Any = None,
    ) -> bool:
        """True if any registered rule fires for this (parent_values, current_val) pair."""
        for rule in self._hard_rules.get(node, []):
            if not all(parent_values.get(k) == v for k, v in rule["when"].items()):
                continue
            if "from_val" in rule and rule["from_val"] != current_val:
                continue
            return True
        return False

    def apply_hard_rule(
        self,
        node: str,
        parent_values: dict[str, Any],
        current_val: Any = None,
    ) -> Any:
        """Return the target value from the first matching rule. Raises LookupError if none."""
        for rule in self._hard_rules.get(node, []):
            if not all(parent_values.get(k) == v for k, v in rule["when"].items()):
                continue
            if "from_val" in rule and rule["from_val"] != current_val:
                continue
            return rule["value"]
        raise LookupError(
            f"No matching hard rule for node='{node}', "
            f"parents={parent_values}, current_val={current_val!r}"
        )

    # ── misc ─────────────────────────────────────────────────────────────────

    def nodes(self) -> list[str]:
        return list(self._g.nodes)

    def edges(self) -> list[tuple[str, str]]:
        return list(self._g.edges)

    def __contains__(self, node: str) -> bool:
        return node in self._g
