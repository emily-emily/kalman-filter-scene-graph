Quantifying distance between scene graphs
==

# Motivation

To quantify the results of any scene graph generating/smoothing algorithm, we need to decide on a quantitative measure of similarity between scene graphs.

There are existing measures for generic graphs such as Graph Edit Distance, however such methods are computationally expensive and do not cater to the specific problem of scene graphs, where nodes may have attributes and labels, and edges have predicates.

# Problem definition

A scene graph has these components:
- Node (object)
    - ID
    - Category label
    - List of attributes
    - Bounding box
- Edge (relationship)
    - Predicate

Considerations
- Restricting words in categories, attributes, predicates to a finite set of possible words
- How much weight to put onto a diff in a certain dimension
    - Eg. we probably care more about a wrong category than a missing attribute
- Scalability: score must be processable in a reasonable amount of time
    - Must be able to evaluate many scene graphs to deal with dynamic scene graphs
- Objects can enter and exit a scene
- Expect to see smoothing between frames, but predicates and attributes may change

# Existing Measures

## Graph Edit Distance (GED)

Description
- Count the number of operations to transform one graph into the other.
- Elementary operations: insert/delete/edit vertex, insert/delete/edit edge.
- Considerations for scene graphs:
    - Complex nodes: a node can have a lot of different things attached to it, including label, bounding box.
        - Maybe represent attributes as a special set of nodes and draw edges between them and object nodes (many to many).
        - Label can be node edit.
        - Bounding box not sure... IoU from 0 to 1?


Advantages

Limitations
- Very slow (see graph in `Graph_Distance.ipynb`)
    - Tested `NetworkX`: `graph_edit_distance`
    - Not feasible for practical use for graphs with more than a few nodes

https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640528.pdf
- Editing scene graphs using GED-based reward

## SPICE

- https://arxiv.org/pdf/1607.08822
- F-score: basically flattens the graph into a set of tuples and finds the F-score of it

## Recall (as used in SAMJAM)

Description
- Criteria
    - Correct labelling of relationship triple (subject, predicate, object)
    - Reasonable bounding box
    - Consistent ID

Advantages
- Relatively simple to compute and understand

Limitations
- Required extensive manual work to track IDs, check bounding boxes, and evaluate things manually
- Possibly can be automated in some ways given human-validated ground truth scene graphs (incl. bounding boxes and labels/predicates)

## Weisfeiler-Lehman Graph Kernel

https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

Description
- Graph similarity for a labelled graph
- For $h$ iterations:
    - Relabel each node with a tuple consisting of itself and all its neighboring nodes
    - Compress labels by hashing
- Construct a vector with frequency of original labels and final labels
    - take the inner product
- Runtime: $O(nm)$ for $h$ iterations and $m$ edges

Advantages
- Fast
- Accounts for node labels

Limitations
- Will need to extend to consider additional complexity of scene graphs

# A custom measure?

What to penalize:
- Categories and predicates changing
- Attributes changing / disappearing / reappearing
- Objects coming in and out of graphs

ID-based measure
- Let $G_1$, $G_2$ be graphs
    - $n_1$, $n_2$ nodes respectively
    - $m_1$, $m_2$ edges respectively
- Take the ratio
- For each pair of nodes with matching IDs
    - Bounding box must have IoU > 0.7 (?)
        - Problem for scenes moving quickly
    - Category must be the same
- Runtime:


next steps
- hierarchical scene graphs - change the format to make the search space smaller

# Hierarchical scene graph (tree) edit distance

Zhang-Shasha (https://www.researchgate.net/publication/220618233_Simple_Fast_Algorithms_for_the_Editing_Distance_Between_Trees_and_Related_Problems)

- Simple tree structure
- Each node is defined to have one label (implemented as a string in `zss`)
- Operations defined:
    - Let A->B->C be a tree.
    - Change a node: change its label
        - A->X->C
    - Insert a node
        - A->X->B->C
    - Delete a node
        - A->C
- Note that this does not enable moving subtrees.


