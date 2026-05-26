# hand_identity_repr_v1

## Research Question
Can contrastive learning with opponent hand identity (J/Q/K) as the supervision signal produce a trainable representation that separates states by opponent hand? What effective dimension does it use?

## Motivation
The existing `contrastive_repr_v1` experiment uses terminal reward as its contrastive signal. While reward correlates with hand strength, it is a noisy proxy: the same hand can win or lose depending on game dynamics, pot odds, and bluffing. This noise means the encoder may learn features correlated with win/loss patterns rather than the opponent's actual hand identity.

Hand-identity supervision provides a cleaner signal: we directly supervise the encoder to produce embeddings that separate states by the opponent's true hand (J, Q, or K). This forces the encoder to learn opponent-discriminative features — specifically, to infer the opponent's private card from observable signals like betting patterns, board texture, and pot size.

Key insight: During self-play training, both players' hands are accessible from the game state, so this is valid supervision even though opponent hand is hidden at inference time. The problem is well-posed because in Leduc Hold'em, betting behavior is strongly correlated with hand strength (K > Q > J).

## Hypothesis
The encoder will achieve >50% opponent hand classification accuracy from its embeddings (chance baseline = 33.3%). Triplet loss with hand labels will converge and produce an embedding space where states with the same opponent hand cluster together.

## Success Criteria
- Linear probe accuracy > 50% (significantly above 33% chance)
- Triplet loss converges (decreases and stabilizes)
- PCA effective dimension (components for 80% variance) < 8 (sparse use of capacity)
- Spearman correlation between embedding distance and hand label distance is positive and significant
