import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Defines the multi-headed self attention mechanism using key, queries and values matrices.
    """
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.to_keys = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.to_queries = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.to_values = nn.Linear(embedding_dim, embedding_dim * heads, bias=False)
        self.unify_heads = nn.Linear(heads * embedding_dim, embedding_dim)

    def forward(self, x):
        """Defines the matrix operations needed for the self-attention section"""
        batch_size, tweet_length, embedding_dim = x.size()
        
        keys = self.to_keys(x).view(batch_size, tweet_length, self.heads, embedding_dim)
        queries = self.to_queries(x).view(batch_size, tweet_length, self.heads, embedding_dim)
        values = self.to_values(x).view(batch_size, tweet_length, self.heads, embedding_dim)
        
        # .view() performs a tensor reshape. If you have already performed a
        # transpose operation, Pytorch needs a .contiguos() before the reshape
        keys = (
            keys.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        queries = (
            queries.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        values = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_size * self.heads, tweet_length, embedding_dim)
        )
        queries = queries / (embedding_dim ** (1 / 4))
        keys = keys / (embedding_dim ** (1 / 4))

        dot = F.softmax(torch.bmm(queries, keys.transpose(1, 2)), dim=2)

        out = torch.bmm(dot, values).view(
            batch_size, self.heads, tweet_length, embedding_dim
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, tweet_length, self.heads * embedding_dim)
        )
        return self.unify_heads(out)


class TransformerBlock(nn.Module):
    """
    Defines the forward network structure of a unique transformer
    block. This is a stack of self-attention, normalisation, 
    fully-connected and dropout layers.
    """
    def __init__(self, embedding_dim, num_heads, *, fc_hidden_multiply=4, dropout=0.4):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * fc_hidden_multiply),
            nn.ReLU(),
            nn.Linear(embedding_dim * fc_hidden_multiply, embedding_dim),
        )
        self.droput = nn.Dropout(dropout)

    def forward(self, x):
        """Defines the basic layer operations within the block."""
        attended = self.attention(x)
        # Skipnet is applied
        x = self.norm1(attended + x)
        x = self.droput(x)
        feedforward = self.fc(x)
        x = self.norm2(feedforward + x)
        x = self.droput(x)
        return x


class Transformer(nn.Module):
    """
    Defines the full structure of a Transformer Neural Network. The main components are:
    1. Token embedding: from sequence of text tokens to sequence of integers.
    2. Positional embedding: vectors representing the position of tokens in a tweet
    3. Transformer blocks: deep learning architecture to be trained.
    4. Conversion of output layer to probabilities: to obtain an answer from model what is the emotion embedded in a given tweet.
    """
    def __init__(
        self,
        embedding_dim,
        seq_length,
        num_heads,
        num_tokens,
        depth,
        num_labels,
        output_dropout=0.2,
        block_dropout=0.3):
        
        super().__init__()
        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=num_tokens
        )
        self.positional_embedding = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=seq_length
        )
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(
                TransformerBlock(embedding_dim, num_heads, dropout=block_dropout)
            )

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.to_probabilities = nn.Linear(embedding_dim, num_labels)
        self.dropout = nn.Dropout(output_dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # size x: (batch_size, tweet_length)
        tokens = self.token_embedding(x)
        # size x: (batch_size, tweet_length, embedding_dim)
        batch_size, tweet_length, embedding_dim = tokens.size()
        positions = torch.unsqueeze(
            self.positional_embedding(torch.arange(tweet_length).to(self.device)), 0
        ).expand(batch_size, tweet_length, embedding_dim)
        x = tokens + positions
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = x.max(dim=1)[0]
        x = self.to_probabilities(x)
        return F.log_softmax(x, dim=1)  # log_softmax instead of softmax as adds further penalty
