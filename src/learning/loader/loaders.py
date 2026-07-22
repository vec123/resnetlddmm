
import torch

class OneBatchLoader:
    """Yields the same prebuilt (graph, true_verts, mask) batch each step."""
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


class ResamplingGraphLoader:
    """Rebuilds the encoder graph from raw geometry each step with a fresh rng.

    Graph construction is delegated to a ``GraphBuilder`` strategy
    (src/learning/data/builders.py): the loader calls
    ``builder.build(verts, mask, rng, areas=..., normals=...)`` and knows nothing
    about radii, dropout, or supernodes -- that policy lives on the ``GraphSpec``
    the builder was constructed with. Swapping radius-graph for kNN construction is
    now "pass a different ``GraphBuilder``", not a change to this class.

    ``vertices`` / ``mask`` / ``areas`` / ``normals`` are direct arguments: they are
    the loader's fixed per-shape geometry (parallel arrays, indexed identically), not a
    construction choice, so they don't belong on the builder/spec.

    The ``rng`` generator's state advances as it is consumed, giving a different graph
    each step while staying reproducible from the seed.

    ``two_view`` makes the contrastive path OPTIONAL: when False (default) each step
    yields the ordinary single-view ``(graph, super, true_verts, mask)`` four-tuple; when
    True it draws TWO independent graphs (different vertex dropout / sampling) of the same
    shapes and yields ``(graph_a, super_a, true_verts, mask, graph_b, super_b)`` -- view A
    in the first four slots so the logger (which reads ``batch[:4]``) is unaffected. The
    six-tuple is what ``TrainingStepper.train_step`` turns into a "same shape -> same
    encoding" alignment loss.

    ``batch_size`` selects mini-batching over SHAPES (dim 0 of ``vertices``): when set and
    smaller than the dataset, each step draws a fresh random subset of that many shapes and
    builds the graph from only those. The yielded ``true_verts`` / ``mask`` are sliced to
    the SAME subset so the decoder's reconstruction target matches the encoder's input; in
    ``two_view`` mode both views use the same subset. ``None`` (default) keeps the original
    full-batch behaviour (every shape, every step).
    """
    def __init__(self, vertices, mask, builder,
                 rng=None, two_view=False, batch_size=None,
                 areas=None, normals=None):

        self.vertices = vertices
        self.mask = mask
        self.builder = builder
        self.rng = rng
        self.two_view = two_view
        self.batch_size = batch_size
        self.areas = areas
        self.normals = normals

    def _pick_indices(self):
        """Draw a fresh random subset of shape indices, or None for the full batch."""
        n = self.vertices.shape[0]
        if self.batch_size is None or self.batch_size >= n:
            return None
        return torch.randperm(n, generator=self.rng)[:self.batch_size]

    def _draw(self, idx=None):
        """Build one fresh view (new dropout / node sampling) from the fixed geometry.

        ``idx`` (if given) restricts the graph to that subset of shapes; the matching
        sliced ``(vertices, mask)`` are returned alongside so the caller can use them as
        the reconstruction target.
        """
        verts = self.vertices if idx is None else self.vertices[idx]
        mask = self.mask if idx is None else self.mask[idx]
        areas = self.areas if (idx is None or self.areas is None) else self.areas[idx]
        normals = self.normals if (idx is None or self.normals is None) else self.normals[idx]

        graph, super_graph = self.builder.build(verts, mask, self.rng,
                                                  areas=areas, normals=normals)
        return graph, super_graph, verts, mask

    def __iter__(self):
        while True:
            idx = self._pick_indices()
            graph_a, super_a, verts, mask = self._draw(idx)
            if self.two_view:
                # Same shape subset, fresh sampling -> "same shape -> same encoding".
                graph_b, super_b, _, _ = self._draw(idx)
                yield (graph_a, super_a, verts, mask, graph_b, super_b)
            else:
                yield (graph_a, super_a, verts, mask)