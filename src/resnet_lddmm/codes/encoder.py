"""Encoder-based codes via GroupEncoder (STEPS T31 T32).

All heavy imports (e3nn, torch_geometric, GroupEncoder) live in this module.
The lazy Registry guarantees that configs using auto_decoder work even without
these dependencies installed.
"""

import torch
from src.resnet_lddmm.codes.base import ShapeCode


class EncoderCodes(ShapeCode):
    """Amortised codes via equivariant graph encoding.

    Builds a graph from point cloud, passes it through GroupEncoder,
    and extracts latent codes. The encoder output also carries pose
    information (rotation/translation) for optional pre-alignment (T32).

    All heavy imports are deferred until instantiation: when this class
    is imported, torch_geometric, e3nn, and GroupEncoder are not loaded.
    """

    def __init__(self, graph_builder, encoder, n_z=None):
        """Initialize encoder-based code source.

        Args:
            graph_builder: GraphBuilder instance (from Registry.create)
            encoder: GroupEncoder instance (from Registry.create)
            n_z: latent dimension (optional, inferred from encoder if not provided)
        """
        super().__init__()
        self.graph_builder = graph_builder
        self.encoder = encoder
        self.n_z = n_z or encoder.latent_dim
        self._last = None  # Stores last EncoderOutput for pose access (T32)

    def forward(self, batch):
        """Extract codes via graph encoding.

        Args:
            batch: CohortBatch with .points [B, N, 3], .shape_ids, optional .weights/.faces

        Returns:
            [B, n_z] codes sampled from encoder's latent distribution
        """
        points = batch.points  # [B, N, 3]
        B = points.shape[0]
        N = points.shape[1]

        # Build per-node mask: shape index for each node
        # e.g., [0,0,...,0,  1,1,...,1,  2,2,...,2] for 3 shapes
        mask = torch.arange(B, device=points.device).repeat_interleave(N)

        # Get RNG for graph building
        rng = self._get_rng()

        # Flatten batch to single point cloud for graph building
        points_flat = points.reshape(B * N, 3)

        # Extract weights and normals if available
        weights = None
        if hasattr(batch, 'weights') and batch.weights is not None:
            weights = batch.weights.reshape(-1)

        normals = None
        if hasattr(batch, 'normals') and batch.normals is not None:
            normals = batch.normals.reshape(-1, 3)

        # Build graph
        graph, supergraph = self.graph_builder.build(
            points_flat, mask, rng,
            areas=weights, normals=normals
        )

        # Move graph to same device as points
        graph = graph.to(points.device)
        if supergraph is not None:
            supergraph = supergraph.to(points.device)

        # Forward through encoder: returns EncoderOutput with latent (or mu) and pose
        encoder_out = self.encoder(graph, supergraph)
        self._last = encoder_out

        # Sample from latent (deterministic=True uses mu if VAE, else latent)
        z = encoder_out.sample(deterministic=True)  # [B, n_z]

        return z

    def penalty(self):
        """No regularisation for encoder codes (unlike AutoDecoderCodes embedding table).

        Returns:
            None
        """
        return None

    def _get_rng(self):
        """Get PRNG key for graph building (jax RNG format).

        Returns:
            jax.random.PRNGKey with shape (2,) and dtype uint32
        """
        import jax
        return jax.random.PRNGKey(0)

    def train(self):
        """Set encoder to training mode."""
        super().train()
        self.encoder.train()

    def eval(self):
        """Set encoder to evaluation mode."""
        super().eval()
        self.encoder.eval()
