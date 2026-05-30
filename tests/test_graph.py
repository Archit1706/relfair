"""Tests for relfair.graph — DependencyGraph."""

import pytest
from relfair.graph import DependencyGraph


def test_from_edges_builds_dag(simple_graph):
    assert "sex" in simple_graph
    assert "honorific" in simple_graph
    assert ("sex", "honorific") in simple_graph.edges()


def test_rejects_cycles():
    G = DependencyGraph.from_edges([("A", "B"), ("B", "C")])
    with pytest.raises(ValueError, match="cycle"):
        G.add_edge("C", "A")


def test_parents(simple_graph):
    assert simple_graph.parents("honorific") == ["sex"]
    assert simple_graph.parents("sex") == []


def test_topological_descendants_order(simple_graph):
    desc = simple_graph.topological_descendants("sex")
    # All three descendants must appear; sex itself must not
    assert set(desc) == {"honorific", "marital", "income_k"}
    assert "sex" not in desc


def test_hard_rule_match(simple_graph):
    assert simple_graph.has_hard_rule("honorific", {"sex": "M"})
    assert simple_graph.has_hard_rule("honorific", {"sex": "F"})
    assert not simple_graph.has_hard_rule("honorific", {"sex": "X"})


def test_hard_rule_apply(simple_graph):
    assert simple_graph.apply_hard_rule("honorific", {"sex": "M"}) == "Mr"
    assert simple_graph.apply_hard_rule("honorific", {"sex": "F"}) == "Ms"


def test_hard_rule_no_match_raises(simple_graph):
    with pytest.raises(LookupError):
        simple_graph.apply_hard_rule("honorific", {"sex": "unknown"})


def test_no_hard_rule_for_soft_node(simple_graph):
    # income_k has no hard rule
    assert not simple_graph.has_hard_rule("income_k", {"sex": "M"})


def test_nodes_returns_all(simple_graph):
    nodes = set(simple_graph.nodes())
    assert nodes == {"sex", "honorific", "marital", "income_k"}


def test_descendants_empty_for_leaf(simple_graph):
    # honorific is a leaf — it has no descendants
    assert simple_graph.topological_descendants("honorific") == []


def test_from_val_rule_fires_only_on_matching_current():
    """from_val rules are conditional transitions — they must NOT fire on other current values."""
    G = DependencyGraph.from_edges([("sex", "relationship")])
    G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",    from_val="Husband")
    G.add_hard_rule("relationship", when={"sex": "Female"}, value="Unmarried", from_val="Unmarried")

    parents = {"sex": "Female"}
    assert G.has_hard_rule("relationship", parents, current_val="Husband")
    assert G.apply_hard_rule("relationship", parents, current_val="Husband") == "Wife"

    # Own-child has no matching from_val → no rule fires
    assert not G.has_hard_rule("relationship", parents, current_val="Own-child")

    # Unmarried stays Unmarried
    assert G.apply_hard_rule("relationship", parents, current_val="Unmarried") == "Unmarried"


def test_from_val_none_rule_is_unconditional():
    """Rules without from_val still fire regardless of current node value."""
    G = DependencyGraph()
    G.add_node("sex")
    G.add_node("honorific")
    G.add_edge("sex", "honorific")
    G.add_hard_rule("honorific", when={"sex": "Female"}, value="Ms")

    # Fires no matter what honorific currently is
    assert G.has_hard_rule("honorific", {"sex": "Female"}, current_val="Mr")
    assert G.has_hard_rule("honorific", {"sex": "Female"}, current_val="Ms")
    assert G.apply_hard_rule("honorific", {"sex": "Female"}, current_val="anything") == "Ms"


def test_single_node_graph():
    G = DependencyGraph()
    G.add_node("age")
    assert "age" in G
    assert G.parents("age") == []
    assert G.topological_descendants("age") == []
