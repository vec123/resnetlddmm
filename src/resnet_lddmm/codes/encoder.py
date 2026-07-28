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
        self._last_graph = None  # Stores last graph for logging/visualization
        self._last_supergraph = None  # Stores last supergraph for logging

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

        # Create 2D boolean mask for graph builder (all valid nodes)
        mask = torch.ones((B, N), dtype=torch.bool, device=points.device)

        # Get RNG for graph building
        rng = self._get_rng()

        # Extract weights and normals if available
        weights = None
        if hasattr(batch, 'weights') and batch.weights is not None:
            weights = batch.weights

        normals = None
        if hasattr(batch, 'normals') and batch.normals is not None:
            normals = batch.normals

        # Build graph (vertices should be [B, N, 3], mask [B, N])
        graph, supergraph = self.graph_builder.build(
            points, mask, rng,
            areas=weights, normals=normals
        )

        # Cache graph for logging/visualization
        self._last_graph = graph
        self._last_supergraph = supergraph

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
        """Get PRNG for graph building.

        Returns:
            None to use default PyTorch RNG
        """
        # Return None to use default RNG; can be seeded via torch.manual_seed if needed
        return None

    def train(self):
        """Set encoder to training mode."""
        super().train()
        self.encoder.train()

    def eval(self):
        """Set encoder to evaluation mode."""
        super().eval()
        self.encoder.eval()

    def get_pose(self):
        """Get encoder's last predicted pose (rotation, translation) for T32 pre-alignment.

        Returns:
            (rotation, translation) tuple where each is [B, ...] or None if no last output
        """
        if self._last is None:
            return None, None
        return self._last.rotation, self._last.translation

    def with_pose_transform(self, base_transform):
        """Create a pose-aware FrameTransform by folding encoder pose into base transform.

        Args:
            base_transform: FrameTransform with center/scale

        Returns:
            New FrameTransform with encoder's rotation/translation folded in, or base_transform if no pose
        """
        if self._last is None or (self._last.rotation is None and self._last.translation is None):
            return base_transform

        from src.resnet_lddmm.io import FrameTransform
        return FrameTransform(
            center=base_transform.center,
            scale=base_transform.scale,
            rotation=self._last.rotation,
            translation=self._last.translation
        )
