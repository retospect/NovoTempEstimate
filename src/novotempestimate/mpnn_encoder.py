"""
MPNN-based protein encoder for temperature prediction.

This module implements a message passing neural network encoder inspired by ProteinMPNN
for protein temperature prediction. It processes 3D structural features and geometric
relationships between residues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import math


@dataclass
class MPNNConfig:
    """Configuration for MPNN encoder."""
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 32
    dropout: float = 0.1
    edge_features_dim: int = 64
    node_features_dim: int = 128
    max_sequence_length: int = 2000
    use_virtual_cb: bool = True
    backbone_noise_std: float = 0.02


class GeometricFeatures:
    """Utility class for computing geometric features from protein coordinates."""
    
    @staticmethod
    def compute_virtual_cb(n_coords: torch.Tensor, ca_coords: torch.Tensor, 
                          c_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute virtual Cβ coordinates from backbone atoms.
        
        Args:
            n_coords: N atom coordinates [batch, seq_len, 3]
            ca_coords: Cα atom coordinates [batch, seq_len, 3]  
            c_coords: C atom coordinates [batch, seq_len, 3]
            
        Returns:
            Virtual Cβ coordinates [batch, seq_len, 3]
        """
        # Vector from Cα to N
        ca_n = n_coords - ca_coords
        # Vector from Cα to C
        ca_c = c_coords - ca_coords
        
        # Normalize vectors
        ca_n = F.normalize(ca_n, dim=-1)
        ca_c = F.normalize(ca_c, dim=-1)
        
        # Compute bisector direction
        bisector = F.normalize(ca_n + ca_c, dim=-1)
        
        # Cross product to get perpendicular direction
        perpendicular = torch.cross(ca_n, ca_c, dim=-1)
        perpendicular = F.normalize(perpendicular, dim=-1)
        
        # Cβ is positioned using tetrahedral geometry
        # Standard Cα-Cβ distance is ~1.54 Å
        cb_direction = F.normalize(-bisector + 0.5 * perpendicular, dim=-1)
        virtual_cb = ca_coords + 1.54 * cb_direction
        
        return virtual_cb
    
    @staticmethod
    def compute_distance_matrix(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix between two sets of coordinates.
        
        Args:
            coords1: First set of coordinates [batch, seq_len, 3]
            coords2: Second set of coordinates [batch, seq_len, 3]
            
        Returns:
            Distance matrix [batch, seq_len, seq_len]
        """
        # Expand dimensions for broadcasting
        coords1_expanded = coords1.unsqueeze(2)  # [batch, seq_len, 1, 3]
        coords2_expanded = coords2.unsqueeze(1)  # [batch, 1, seq_len, 3]
        
        # Compute distances
        distances = torch.norm(coords1_expanded - coords2_expanded, dim=-1)
        return distances
    
    @staticmethod
    def get_k_nearest_neighbors(distance_matrix: torch.Tensor, k: int, 
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get k nearest neighbors for each residue.
        
        Args:
            distance_matrix: Pairwise distances [batch, seq_len, seq_len]
            k: Number of nearest neighbors
            mask: Optional mask for valid positions [batch, seq_len]
            
        Returns:
            Neighbor indices [batch, seq_len, k]
        """
        batch_size, seq_len, _ = distance_matrix.shape
        
        if mask is not None:
            # Set masked positions to large distance
            distance_matrix = distance_matrix.clone()
            mask_expanded = mask.unsqueeze(1) & mask.unsqueeze(2)
            distance_matrix[~mask_expanded] = float('inf')
        
        # Get k+1 nearest (including self) then remove self
        _, indices = torch.topk(distance_matrix, k + 1, dim=-1, largest=False)
        
        # Remove self-connections (distance 0)
        neighbor_indices = indices[:, :, 1:k+1]
        
        return neighbor_indices


class MPNNNodeEncoder(nn.Module):
    """Node encoder for MPNN that processes residue-level features."""
    
    def __init__(self, config: MPNNConfig):
        super().__init__()
        self.config = config
        
        # Amino acid embedding (20 standard + 1 unknown)
        self.aa_embedding = nn.Embedding(21, config.node_features_dim // 4)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(config.max_sequence_length, config.node_features_dim // 4)
        
        # Geometric feature projection
        self.geom_projection = nn.Linear(15, config.node_features_dim // 2)  # 5 atoms * 3 coords
        
        # Final node feature projection
        self.node_projection = nn.Linear(config.node_features_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, aa_sequence: torch.Tensor, coordinates: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode node features from sequence and coordinates.
        
        Args:
            aa_sequence: Amino acid sequence [batch, seq_len]
            coordinates: Backbone coordinates [batch, seq_len, 4, 3] (N, CA, C, O)
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Node features [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len = aa_sequence.shape
        device = aa_sequence.device
        
        # Amino acid embeddings
        aa_emb = self.aa_embedding(aa_sequence)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Geometric features from coordinates
        # Flatten coordinates: [batch, seq_len, 4*3] -> [batch, seq_len, 12]
        geom_features = coordinates.view(batch_size, seq_len, -1)
        
        # Add virtual Cβ if enabled
        if self.config.use_virtual_cb:
            n_coords = coordinates[:, :, 0]  # N atoms
            ca_coords = coordinates[:, :, 1]  # Cα atoms  
            c_coords = coordinates[:, :, 2]   # C atoms
            
            virtual_cb = GeometricFeatures.compute_virtual_cb(n_coords, ca_coords, c_coords)
            geom_features = torch.cat([geom_features, virtual_cb], dim=-1)  # [batch, seq_len, 15]
        
        geom_emb = self.geom_projection(geom_features)
        
        # Combine all features
        node_features = torch.cat([aa_emb, pos_emb, geom_emb], dim=-1)
        node_features = self.node_projection(node_features)
        node_features = self.dropout(node_features)
        
        # Apply mask if provided
        if mask is not None:
            node_features = node_features * mask.unsqueeze(-1)
        
        return node_features


class MPNNEdgeEncoder(nn.Module):
    """Edge encoder for MPNN that processes pairwise geometric relationships."""
    
    def __init__(self, config: MPNNConfig):
        super().__init__()
        self.config = config
        
        # Distance feature bins (similar to ProteinMPNN)
        self.distance_bins = torch.linspace(2.0, 20.0, 37)  # 2-20Å in 0.5Å bins
        
        # Edge feature projection
        # 5 distance types * 37 bins + relative position encoding
        edge_input_dim = 5 * 37 + 64  # Distance features + positional
        self.edge_projection = nn.Linear(edge_input_dim, config.edge_features_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, coordinates: torch.Tensor, neighbor_indices: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode edge features from geometric relationships.
        
        Args:
            coordinates: Backbone coordinates [batch, seq_len, 4, 3] (N, CA, C, O)
            neighbor_indices: Neighbor indices [batch, seq_len, k]
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Edge features [batch, seq_len, k, edge_features_dim]
        """
        batch_size, seq_len, k = neighbor_indices.shape
        device = coordinates.device
        
        # Compute virtual Cβ
        if self.config.use_virtual_cb:
            n_coords = coordinates[:, :, 0]
            ca_coords = coordinates[:, :, 1]
            c_coords = coordinates[:, :, 2]
            virtual_cb = GeometricFeatures.compute_virtual_cb(n_coords, ca_coords, c_coords)
            all_coords = torch.cat([coordinates, virtual_cb.unsqueeze(2)], dim=2)  # [batch, seq_len, 5, 3]
        else:
            all_coords = coordinates  # [batch, seq_len, 4, 3]
        
        num_atoms = all_coords.shape[2]
        
        # Compute distance features for all atom pairs
        distance_features = []
        
        for atom_i in range(num_atoms):
            for atom_j in range(num_atoms):
                # Get coordinates for atom types i and j
                coords_i = all_coords[:, :, atom_i]  # [batch, seq_len, 3]
                coords_j = all_coords[:, :, atom_j]  # [batch, seq_len, 3]
                
                # Compute distance matrix
                dist_matrix = GeometricFeatures.compute_distance_matrix(coords_i, coords_j)
                
                # Gather distances for neighbors
                batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
                seq_indices = torch.arange(seq_len, device=device).view(1, -1, 1)
                neighbor_distances = dist_matrix[batch_indices, seq_indices, neighbor_indices]
                
                # Bin distances
                distance_bins = self.distance_bins.to(device)
                binned_distances = torch.bucketize(neighbor_distances, distance_bins)
                binned_distances = torch.clamp(binned_distances, 0, len(distance_bins) - 1)
                
                # Convert to one-hot
                distance_onehot = F.one_hot(binned_distances, num_classes=len(distance_bins))
                distance_features.append(distance_onehot.float())
        
        # Concatenate all distance features
        edge_distances = torch.cat(distance_features, dim=-1)  # [batch, seq_len, k, 5*5*37]
        
        # Add relative positional encoding
        seq_positions = torch.arange(seq_len, device=device).view(1, -1, 1)
        neighbor_positions = neighbor_indices  # [batch, seq_len, k]
        relative_positions = neighbor_positions - seq_positions
        
        # Encode relative positions
        rel_pos_encoding = self._encode_relative_positions(relative_positions)
        
        # Combine distance and positional features
        edge_features = torch.cat([edge_distances, rel_pos_encoding], dim=-1)
        edge_features = self.edge_projection(edge_features)
        edge_features = self.dropout(edge_features)
        
        # Apply mask if provided
        if mask is not None:
            edge_mask = mask.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]
            edge_features = edge_features * edge_mask
        
        return edge_features
    
    def _encode_relative_positions(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """Encode relative positions using sinusoidal encoding."""
        # Clamp to reasonable range
        relative_positions = torch.clamp(relative_positions, -32, 32)
        
        # Sinusoidal encoding
        device = relative_positions.device
        d_model = 64
        
        pe = torch.zeros(*relative_positions.shape, d_model, device=device)
        position = relative_positions.unsqueeze(-1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        
        return pe


class MPNNLayer(nn.Module):
    """Single message passing layer."""
    
    def __init__(self, config: MPNNConfig):
        super().__init__()
        self.config = config
        
        # Node update networks
        self.node_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim + config.edge_features_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # Edge update networks  
        self.edge_mlp = nn.Sequential(
            nn.Linear(config.edge_features_dim + 2 * config.hidden_dim, config.edge_features_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_features_dim * 2, config.edge_features_dim)
        )
        
        self.layer_norm_node = nn.LayerNorm(config.hidden_dim)
        self.layer_norm_edge = nn.LayerNorm(config.edge_features_dim)
        
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor,
                neighbor_indices: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one round of message passing.
        
        Args:
            node_features: Node features [batch, seq_len, hidden_dim]
            edge_features: Edge features [batch, seq_len, k, edge_features_dim]
            neighbor_indices: Neighbor indices [batch, seq_len, k]
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Updated node and edge features
        """
        batch_size, seq_len, k = neighbor_indices.shape
        device = node_features.device
        
        # Gather neighbor node features
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
        seq_indices = torch.arange(seq_len, device=device).view(1, -1, 1)
        neighbor_features = node_features[batch_indices, neighbor_indices]  # [batch, seq_len, k, hidden_dim]
        
        # Update edges
        center_features = node_features.unsqueeze(2).expand(-1, -1, k, -1)  # [batch, seq_len, k, hidden_dim]
        edge_input = torch.cat([edge_features, center_features, neighbor_features], dim=-1)
        edge_updates = self.edge_mlp(edge_input)
        edge_features_new = self.layer_norm_edge(edge_features + edge_updates)
        
        # Aggregate messages from neighbors
        messages = torch.cat([neighbor_features, edge_features_new], dim=-1)  # [batch, seq_len, k, hidden_dim + edge_dim]
        
        # Sum messages from all neighbors
        aggregated_messages = torch.sum(messages, dim=2)  # [batch, seq_len, hidden_dim + edge_dim]
        
        # Update nodes
        node_updates = self.node_mlp(aggregated_messages)
        node_features_new = self.layer_norm_node(node_features + node_updates)
        
        # Apply mask
        if mask is not None:
            node_features_new = node_features_new * mask.unsqueeze(-1)
            edge_features_new = edge_features_new * mask.unsqueeze(-1).unsqueeze(-1)
        
        return node_features_new, edge_features_new


class ProteinMPNNEncoder(nn.Module):
    """
    ProteinMPNN-inspired encoder for protein temperature prediction.
    
    This encoder processes 3D protein structures using message passing neural networks
    to capture geometric relationships and local structural environments.
    """
    
    def __init__(self, config: MPNNConfig):
        super().__init__()
        self.config = config
        
        # Encoders
        self.node_encoder = MPNNNodeEncoder(config)
        self.edge_encoder = MPNNEdgeEncoder(config)
        
        # Message passing layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(config) for _ in range(config.num_encoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, aa_sequence: torch.Tensor, coordinates: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode protein structure using MPNN.
        
        Args:
            aa_sequence: Amino acid sequence [batch, seq_len]
            coordinates: Backbone coordinates [batch, seq_len, 4, 3] (N, CA, C, O)
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Encoded features [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len = aa_sequence.shape
        
        # Add backbone noise during training for robustness
        if self.training and self.config.backbone_noise_std > 0:
            noise = torch.randn_like(coordinates) * self.config.backbone_noise_std
            coordinates = coordinates + noise
        
        # Compute k-nearest neighbors based on Cα distances
        ca_coords = coordinates[:, :, 1]  # Cα atoms
        ca_distances = GeometricFeatures.compute_distance_matrix(ca_coords, ca_coords)
        neighbor_indices = GeometricFeatures.get_k_nearest_neighbors(
            ca_distances, self.config.num_neighbors, mask
        )
        
        # Encode initial node and edge features
        node_features = self.node_encoder(aa_sequence, coordinates, mask)
        edge_features = self.edge_encoder(coordinates, neighbor_indices, mask)
        
        # Message passing
        for mpnn_layer in self.mpnn_layers:
            node_features, edge_features = mpnn_layer(
                node_features, edge_features, neighbor_indices, mask
            )
        
        # Final projection
        encoded_features = self.output_projection(node_features)
        
        return encoded_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ProteinMPNN_Encoder',
            'hidden_dim': self.config.hidden_dim,
            'num_encoder_layers': self.config.num_encoder_layers,
            'num_neighbors': self.config.num_neighbors,
            'edge_features_dim': self.config.edge_features_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': next(self.parameters()).device.type
        }


def create_mpnn_encoder(hidden_dim: int = 128, num_layers: int = 3, 
                       num_neighbors: int = 32, **kwargs) -> ProteinMPNNEncoder:
    """Create a ProteinMPNN encoder with specified configuration."""
    config = MPNNConfig(
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_neighbors=num_neighbors,
        **kwargs
    )
    return ProteinMPNNEncoder(config)
