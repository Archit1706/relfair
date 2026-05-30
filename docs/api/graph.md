# `relfair.graph`

## `DependencyGraph`

```python
from relfair.graph import DependencyGraph
```

### Construction

```python
G = DependencyGraph()                          # empty graph
G = DependencyGraph.from_edges([               # from edge list
    ("sex", "relationship"),
    ("age", "occupation"),
])
```

### Core methods

| Method | Returns | Description |
|---|---|---|
| `G.add_edge(parent, child)` | `None` | Add a causal edge. Raises `ValueError` if it creates a cycle. |
| `G.add_hard_rule(node, when, value, from_val=None)` | `None` | Add a hard rule. |
| `G.nodes()` | `list[str]` | All node names. |
| `G.parents(node)` | `list[str]` | Direct parents of a node. |
| `G.descendants(node)` | `list[str]` | All transitive descendants in topological order. |
| `G.has_hard_rule(node, parent_vals, current_val=None)` | `bool` | Whether a hard rule fires. |
| `G.apply_hard_rule(node, parent_vals, current_val=None)` | `Any` | Apply the matching hard rule. |
| `G.topological_order()` | `list[str]` | All nodes in topological sort order. |

### Hard rule semantics

```python
# Unconditional: fires for any current value of the node
G.add_hard_rule("honorific", when={"sex": "Female"}, value="Ms")

# Conditional transition: only fires when current value == from_val
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife", from_val="Husband")

# Passthrough: preserve values that should not change
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")
```

!!! danger "Always add passthrough rules"
    A rule without `from_val` rewrites **all** current values. Add an explicit passthrough
    for every value that should remain unchanged under the intervention.
