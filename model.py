import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the token embeddings using sine and cosine functions.
    """
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to input tensor x.
        Args:
            x: Tensor of shape [batch_size, seq_length, embed_dim]
        Returns:
            Tensor with positional encoding added: [batch_size, seq_length, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.
    """
    def __init__(self, hidden_dim):
        super(ResidualConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, sublayer):
        """
        Applies a residual connection to the output of a sublayer.
        Args:
            x: Original input tensor
            sublayer: Function representing the sublayer (e.g., attention or feed-forward)
        Returns:
            Tensor after applying residual connection and layer normalization
        """
        return self.layer_norm(x + sublayer(x))

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)  

    def forward(self, queries, keys, values, mask):
        batch_size = queries.size(0)

        Q = self.query(queries)  # [batch_size, seq_length, hidden_dim]
        K = self.key(keys)
        V = self.value(values)

        # split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_length, seq_length]

        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)  # [batch_size, num_heads, seq_length, seq_length]
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)  # [batch_size, num_heads, seq_length, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)  # [batch_size, seq_length, hidden_dim]

        out = self.fc_out(out)  # [batch_size, seq_length, hidden_dim]
        return out

class FeedForward(nn.Module):
    """
    Implements a simple feed-forward network.
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        Args:
            x: Input tensor [batch_size, seq_length, hidden_dim]
        Returns:
            Output tensor after feed-forward network
        """
        residual = x
        x = self.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.layer_norm(x + residual)
        return x

class Attention(nn.Module):
    """
    Computes a context vector that emphasizes relevant parts of the LSTM outputs for classification.
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(Attention, self).__init__()
        self.multihead_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.residual = ResidualConnection(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, dropout)

    def forward(self, encoder_outputs, mask):
        """
        Applies multi-head attention followed by residual connection and feed-forward network.
        Args:
            encoder_outputs: [batch_size, seq_length, hidden_dim]
            mask: [batch_size, seq_length]
        Returns:
            context: [batch_size, hidden_dim]
        """
        attn_output = self.multihead_attn(encoder_outputs, encoder_outputs, encoder_outputs, mask)  # [batch_size, seq_length, hidden_dim]
        attn_output = self.residual(encoder_outputs, lambda x: attn_output)  # [batch_size, seq_length, hidden_dim]
        attn_output = self.feed_forward(attn_output)  # [batch_size, seq_length, hidden_dim]

        # global average pooling to get context vector
        context = attn_output.mean(dim=1)  # [batch_size, hidden_dim]
        return context

class ThemeClassifierSLM(nn.Module):
    """
    An LSTM-based model for theme classification, enhanced with positional encoding, residual connections,
    multi-head attention, and regularization techniques.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pretrained_embeddings=None, max_len=5000, num_heads=8):
        super(ThemeClassifierSLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # freezing the embeddings to prevent updates
        
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim, num_heads, dropout)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        Returns:
            logits: [batch_size, output_dim]
        """
        embedded = self.embedding(input_ids)  # [batch_size, seq_length, embed_dim]
        embedded = self.positional_encoding(embedded)  # [batch_size, seq_length, embed_dim]
        embedded = self.dropout(embedded)  # regularization
        
        lengths = attention_mask.sum(dim=1)  # [batch_size]
        
        # pack the padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                            batch_first=True, enforce_sorted=False)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack the sequences
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=attention_mask.size(1))
        # encoder_outputs: [batch_size, seq_length, hidden_dim * num_directions]
        
        # attention
        context = self.attention(encoder_outputs, attention_mask)  # [batch_size, hidden_dim * num_directions]
        
        # layer normalization
        context = self.layer_norm(context)
        
        # fully connected layer
        logits = self.fc(context)  # [batch_size, output_dim]
        
        return logits

def get_model(vocab_size, embed_dim=100, hidden_dim=256, output_dim=20, n_layers=2, 
             bidirectional=True, dropout=0.5, pretrained_embeddings=None, max_len=5000, num_heads=8):
    """
    Utility function to initialize the ThemeClassifierSLM.
    Ensures that hidden_dim is divisible by num_heads.
    """
    assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
    model = ThemeClassifierSLM(vocab_size, embed_dim, hidden_dim, output_dim, 
                               n_layers, bidirectional, dropout, pretrained_embeddings, max_len, num_heads)
    return model
