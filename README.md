# Transformer

A from-scratch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Each component : attention, positional encoding, feed-forward layers, encoder and decoder stacks  is implemented as a separate module.


## Modules

| Module | Description |
|---|---|
| `scaled_dot_product.py` | Computes `softmax(QKᵀ / √dₖ) V` |
| `MultiHeadAttention.py` | Splits Q, K, V into multiple heads, applies attention in parallel, concatenates results |
| `PositionalEncoding.py` | Injects sequence-order information using sine/cosine functions |
| `FeedForwardNetwork.py` | Two-layer fully connected network applied to each position |
| `EncoderLayer.py` | Self-attention + feed-forward, with residual connections and layer norm |
| `Encoder.py` | Stacks `N` encoder layers |
| `DecoderLayer.py` | Masked self-attention + cross-attention + feed-forward |
| `Decoder.py` | Stacks `N` decoder layers |
| `Transformer.py` | Combines encoder, decoder, embeddings, and output projection |

##Feedback
I welcome feedback and suggestions for improving this repository. Please feel free to reach out to me via email or GitHub issues.


## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*.

## License

MIT

