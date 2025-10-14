from typing import Tuple, Union, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bart.configuration_bart import BartConfig
from .mlp import MLP

from src.loader.data_format.processors.subprocessors import ProcessedMonomial


class ExponentEmbedding(nn.Embedding):
    """Embedding layer for exponent vectors."""
    def __init__(self, 
                num_variables: int,
                max_degree: int,
                embedding_dim: int,
                ):
        super().__init__(num_variables*(max_degree+1), embedding_dim)
        
        shift = torch.arange(num_variables).long() * (max_degree + 1)
        self.register_buffer('shift', shift)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert exponent vectors to embedding vectors.
        
        Args:
            x: (batch_size, seq_length, num_variables) - Exponent vectors
        Returns:
            embeddings: (batch_size, seq_length, num_variables, embedding_dim)
        """        
        z = (x + self.shift).view(x.shape[0], -1)
        z = super().forward(z)
        z = z.view(x.shape[0], x.shape[1], x.shape[2], -1)

        return z.sum(dim=-2)
    
    
class ContinuousCoefficientEmbedding(nn.Module):
    """Embedding layer for continuous coefficients with option for linear or MLP projection."""
    def __init__(self, 
                 embedding_dim: int,
                 use_mlp: bool = True,
                 hidden_dim: int = None,
                 num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_mlp = use_mlp
        
        if use_mlp:
            if hidden_dim is None:
                hidden_dim = embedding_dim * 2
            
            layers = []
            # Input layer
            layers.append(nn.Linear(1, hidden_dim))
            layers.append(nn.ReLU())
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, embedding_dim))
            
            self.projection = nn.Sequential(*layers)
        else:
            # Original linear projection for backward compatibility
            self.projection = nn.Linear(1, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project continuous coefficients to embedding space.
        
        Args:
            x: (batch_size, seq_length) - Continuous coefficient values
        Returns:
            embeddings: (batch_size, seq_length, embedding_dim)
        """
        # Add channel dimension for projection
        x = x.unsqueeze(-1)
        return self.projection(x)

class TokenExpander(nn.Module):
    """Sophisticated token expansion module that uses attention and MLP to expand hidden states into multiple tokens.
    
    This module uses a combination of:
    1. Self-attention to capture dependencies between tokens
    2. MLP for token-specific transformations
    3. Residual connections for better gradient flow
    """
    def __init__(self, 
                 d_model: int,
                 tokens_per_unit: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_mlp: bool = True):
        super().__init__()
        self.d_model = d_model
        self.tokens_per_unit = tokens_per_unit
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        
        # Token-specific projections
        self.token_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(tokens_per_unit)
        ])
        
        # Self-attention for token interactions
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm for attention output
        self.norm1 = nn.LayerNorm(d_model)
        
        # MLP for additional processing
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)
        
        # Final projection to get all tokens
        self.final_projection = nn.Linear(d_model, d_model * tokens_per_unit)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand hidden states into multiple tokens using attention and MLP.
        
        Args:
            hidden_states: (batch_size, seq_length, d_model)
        Returns:
            expanded_states: (batch_size, seq_length * tokens_per_unit, d_model)
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project each token position
        token_states = []
        for i in range(self.tokens_per_unit):
            token_states.append(self.token_projections[i](hidden_states))
        token_states = torch.stack(token_states, dim=1)  # (batch, tokens_per_unit, seq, d_model)
        
        # Reshape for attention
        token_states = token_states.view(batch_size * self.tokens_per_unit, seq_length, self.d_model)
        
        # Apply self-attention
        attn_output, _ = self.self_attn(token_states, token_states, token_states)
        attn_output = self.norm1(attn_output + token_states)  # Residual connection
        
        # Apply MLP if enabled
        if self.use_mlp:
            mlp_output = self.mlp(attn_output)
            attn_output = self.norm2(attn_output + mlp_output)  # Residual connection
        
        # Reshape back
        attn_output = attn_output.view(batch_size, self.tokens_per_unit, seq_length, self.d_model)
        
        # Combine token states
        combined_states = attn_output.sum(dim=1)  # (batch, seq, d_model)
        
        # Final projection to get all tokens
        expanded = self.final_projection(combined_states)  # (batch, seq, d_model * tokens_per_unit)
        
        # Reshape to final format
        expanded = expanded.view(batch_size, seq_length, self.tokens_per_unit, self.d_model)
        expanded = expanded.view(batch_size, -1, self.d_model)
        
        return expanded

class MonomialEmbedding(nn.Module):
    """Embedding layer for monomials.
    
    Takes monomial representations (coefficient and exponents) and converts them to embeddings.
    Can also expand hidden states for decoding.
    """
    def __init__(self, 
                config: BartConfig, 
                num_coefficients: int, 
                max_degree: int,
                num_variables: int,
                token_expander: str = 'mlp1',
                rational_coefficients: List[Tuple[int, int]] = None,
                continuous_coefficient: bool = False,
                use_advanced_expander: bool = False,
                expander_type: str = "linear"):
        super().__init__()
        self.d_model = config.d_model
        self.num_variables = num_variables
        self.tokens_per_unit = num_variables + 2  # coefficient + exponents + operator
        self.continuous_coefficient = continuous_coefficient
        
        # Calculate total number of coefficients including rationals
        total_coefficients = num_coefficients + 1  # Original coefficients (0 to num_coefficients)
        if rational_coefficients:
            total_coefficients += len(rational_coefficients)  # Add space for rational coefficients
        
        # For encoding
        self.coef_embeddings = nn.Embedding(total_coefficients, self.d_model)
        if continuous_coefficient:
            self.continuous_coef_embedding = ContinuousCoefficientEmbedding(self.d_model)
        self.exponent_embeddings = ExponentEmbedding(num_variables, max_degree, self.d_model)
        self.sepcial_embedding = nn.Embedding(10, self.d_model)
        
        # Initialize token expander based on configuration
        if expander_type == "mlp1" or expander_type == "mlp2":
            print("Using advanced token expander")
            self.token_expander = TokenExpander(
                d_model=self.d_model,
                tokens_per_unit=self.tokens_per_unit,
                num_heads=4,  # Can be made configurable
                dropout=0.1,  # Can be made configurable
                use_mlp=True
            )
        else:
            print("Using simple linear token expander")
            # Original simple linear expansion for backward compatibility
            self.token_expander = nn.Linear(self.d_model, self.d_model * self.tokens_per_unit)

    def encode(self, monomial_ids: Union[torch.Tensor, List[ProcessedMonomial]], coefficient_values: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """Convert monomial ID sequence to embedding vectors.
        
        Args:
            monomial_ids: Either:
                - (batch_size, seq_length, num_variables + 2) tensor of discrete IDs
                - List of ProcessedMonomial for continuous coefficients
            coefficient_values: Optional tensor of continuous coefficient values
                Only used when monomial_ids is a tensor and continuous_coefficient=True
        Returns:
            embeddings: (batch_size, seq_length, d_model)
        """
        if isinstance(monomial_ids, torch.Tensor):
            # Original discrete mode
            coef_ids = monomial_ids[..., 0]
            exponent_ids = monomial_ids[..., 1:-1]
            special_ids = monomial_ids[..., -1]
            
            assert coef_ids.max().item() < self.coef_embeddings.num_embeddings, \
                f"coef_ids has out-of-bounds index {coef_ids.max().item()} (max allowed: {self.coef_embeddings.num_embeddings - 1})"
            assert exponent_ids.max().item() < self.exponent_embeddings.num_embeddings, \
                f"exponent_ids has out-of-bounds index {exponent_ids.max().item()} (max allowed: {self.exponent_embeddings.num_embeddings - 1})"
            assert special_ids.max().item() < self.sepcial_embedding.num_embeddings, \
                f"special_ids has out-of-bounds index {special_ids.max().item()} (max allowed: {self.sepcial_embedding.num_embeddings - 1})"

            if self.continuous_coefficient and coefficient_values is not None:
                # Use continuous coefficient embeddings
                coef_embeddings = self.continuous_coef_embedding(coefficient_values)
            else:
                # Use discrete coefficient embeddings
                coef_embeddings = self.coef_embeddings(coef_ids)
            
        else:
            # Continuous mode
            assert self.continuous_coefficient, "Continuous coefficient embedding layer not initialized"
            
            # Convert list of ProcessedMonomial to tensors
            device = next(self.parameters()).device
            batch_size = len(monomial_ids)
            seq_length = len(monomial_ids[0])
            
            # Extract tokens and coefficient values
            tokens = torch.tensor([[m.tokens for m in seq] for seq in monomial_ids], device=device)
            coef_values = torch.tensor([[m.coefficient_value if m.coefficient_value is not None else 0.0 
                                       for m in seq] for seq in monomial_ids], device=device)
            
            exponent_ids = tokens[..., 1:-1]
            special_ids = tokens[..., -1]
            
            # Get continuous coefficient embeddings
            coef_embeddings = self.continuous_coef_embedding(coef_values)

        # Common embedding computation for both modes
        exponent_embeddings = self.exponent_embeddings(exponent_ids)
        special_embeddings = self.sepcial_embedding(special_ids)
        
        embeddings = coef_embeddings + exponent_embeddings + special_embeddings
        return embeddings
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand hidden states into n+2 token hidden states.
        
        Args:
            hidden_states: (batch_size, seq_length, d_model)
        Returns:
            expanded_states: (batch_size, seq_length * (num_variables + 2), d_model)
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        """
            IMPORTANT: Why hidden_states[:, :-1]? 
            MonomialProcesssor appends the [bos] monomial token. This is equivalent to adding (n+2) infix tokens. 
            
            [C] [E] ... [E] [S] ([C]: coefficient, [E]: exponent, [S]: special token (e.g [EOS], [SEP], ...))
            
            decoder input (infix token space)   :   1  n+2  n+2  n+2  ...  n+2 -- the first token is [BOS].
            decoder input (monomial token space): [Mb] [M1] [M2] ... [ML] -- (L+1 tokens, 1 -> [Mb] (equivalent to n+2 tokens))
            decoder output ( ... )              : [M1] [M2] ... [ML] [Me] -- (L+1 tokens)
            labels (monomial token space)       : [A1] [A2] ... [AL] [Ae] -- (L+1 tokens)
            labels (infix token space)          : n+2  n+2  ... n+2   0   -- (L+1 tokens, [Ae] -> 0, as [AL] can include [EOS] information)
            
        """
        # Expand each token into n+2 tokens
        expanded = self.token_expander(hidden_states[:, :-1])  # (batch, seq, d_model * (n+2))


        # Reshape
        expanded = expanded.view(
            batch_size,
            -1,
            self.tokens_per_unit,
            self.d_model
        )
        
        # Expand along sequence dimension
        expanded = expanded.view(
            batch_size,
            -1,
            self.d_model
        )
        
        return expanded
    
    def forward(self, x: Union[torch.Tensor, List[ProcessedMonomial]], mode: str = 'encode', coefficient_values: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """Unified interface for encoding/decoding.
        
        Args:
            x: Input tensor or list of ProcessedMonomial
            mode: 'encode' or 'decode'
            coefficient_values: Optional tensor of continuous coefficient values
                Only used when x is a tensor and continuous_coefficient=True
        Returns:
            embeddings: (batch_size, seq_length, d_model) for encode mode
                       (batch_size, seq_length * (num_variables + 2), d_model) for decode mode
        """
        if mode == 'encode':
            return self.encode(x, coefficient_values=coefficient_values)
        elif mode == 'decode':
            return self.decode(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")