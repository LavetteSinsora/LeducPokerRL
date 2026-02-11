======================================================================
TRAINING TARGET & NORMALIZATION ANALYSIS REPORT
======================================================================
Analyzing trained model: models/value_based_agent.pt
Model loaded. Input size: 14, Temperature: 1.0

--- Data Collection Phase ---
Playing 50,000 self-play episodes with Boltzmann exploration...
  Collected 10000/50000 episodes...
  Collected 20000/50000 episodes...
  Collected 30000/50000 episodes...
  Collected 40000/50000 episodes...
  Collected 50000/50000 episodes...
Collected 196612 decision records from 50,000 episodes.

======================================================================
SECTION A: REWARD DISTRIBUTION ANALYSIS
======================================================================

Total episodes: 50000
Mean reward:    +0.146
Std reward:     5.742
Min reward:     -13
Max reward:     +13
Variance:       32.974

Reward Histogram (from player-0 perspective of first decision):
 Value    Count     Pct  Bar
   -13      874    1.7%  ###
   -11     1090    2.2%  ####
    -9     2529    5.1%  ##########
    -7     2109    4.2%  ########
    -5     3005    6.0%  ############
    -3     3134    6.3%  ############
    -1    14063   28.1%  ########################################################
    +0     6226   12.5%  ########################
    +1     4456    8.9%  #################
    +3     1136    2.3%  ####
    +5     1930    3.9%  #######
    +7     2882    5.8%  ###########
    +9     3027    6.1%  ############
   +11     2143    4.3%  ########
   +13     1396    2.8%  #####

Breakdown by game-ending type:
  preflop_fold   : n= 15281, mean=-0.66, std=1.03, range=[-3, +3]
  flop_fold      : n=  3479, mean=-0.79, std=1.64, range=[-7, +9]
  showdown       : n= 31240, mean=+0.64, std=7.16, range=[-13, +13]

======================================================================
SECTION B: PER-STATE VARIANCE ANALYSIS (CORE EXPERIMENT)
======================================================================

Total unique state keys: 162
Total decision records:  196612

*** WEIGHTED AVERAGE PER-STATE VARIANCE: 31.57 ***
*** Observed training loss:              ~40 ***
*** CONCLUSION: The loss IS approximately the irreducible MC variance! ***

Top 20 highest-variance state keys:
 Card Board Rnd  CP          Pot      n    Mean    Std      Var        Range
    K     J   1   1      (13, 9)     78   -0.67  12.81   164.22 [-13,+13]
    K     J   1   0      (9, 13)    327   +3.22  12.14   147.26 [-13,+13]
    K     J   1   1      (11, 7)    109   +2.62  10.42   108.56 [-11,+11]
    K     J   1   0      (7, 11)    654   +3.74   9.86    97.28 [-11,+11]
    K     Q   1   0      (9, 13)    270   +9.20   8.66    75.04 [-13,+13]
    Q  None   0   0       (3, 5)   2875   +0.40   8.49    72.13 [-13,+13]
    K     J   1   0       (5, 5)   1251   +2.25   8.45    71.36 [-13,+13]
    Q  None   0   1       (5, 3)    967   +2.57   8.35    69.68 [-13,+13]
    K     J   1   0       (5, 9)    700   +2.59   8.32    69.21 [-13,+13]
    K     J   1   1       (9, 5)    743   +2.63   8.16    66.52 [-13,+13]
    Q     K   1   1       (9, 5)    725   +3.53   8.12    65.88 [-13,+13]
    K     Q   1   0       (5, 5)   1232   +3.10   7.88    62.04 [-13,+13]
    K  None   0   0       (3, 5)   2985   +3.92   7.86    61.78 [-13,+13]
    Q     K   1   0       (5, 5)   1439   +3.59   7.83    61.29 [-13,+13]
    J  None   0   1       (3, 1)   6465   -3.26   7.66    58.61 [-13,+13]
    K     Q   1   1       (9, 5)    766   +3.37   7.60    57.77 [-13,+13]
    Q     K   1   1       (7, 3)    738   +0.49   7.10    50.41 [-11,+11]
    J  None   0   1       (5, 3)    767   -2.40   7.07    50.04 [-13,+13]
    Q  None   0   1       (3, 1)   6433   -0.54   7.06    49.83 [-13,+13]
    Q  None   0   0       (1, 3)   3087   +0.15   6.93    48.02 [-13,+13]

Variance breakdown by round:
  Round 0:   18 unique states, weighted var = 36.60, total samples = 111411
  Round 1:  144 unique states, weighted var = 25.00, total samples = 85201

Variance breakdown by private card (round 0 only):
  J:   6 unique states, weighted var = 35.32, total samples = 37070
  Q:   6 unique states, weighted var = 41.57, total samples = 36785
  K:   6 unique states, weighted var = 33.01, total samples = 37556

======================================================================
SECTION C: VALUE NETWORK OUTPUT ANALYSIS
======================================================================

--- Hand Strength Ordering Check (Round 0, pot=[1,1]) ---
If network learned poker: V(K) > V(Q) > V(J) for same position

  Player 0: K(+0.62) > Q(-0.86) > J(-1.93)  CORRECT
  Player 1: K(+0.63) > Q(-0.48) > J(-1.33)  CORRECT

--- Pair vs No-Pair Check (Round 1) ---
If network learned poker: V(pair) > V(high card no pair)

  CP=0, Board=J: K(hi)=+2.36, Q(hi)=-0.24, J(PAIR)=-1.67  WRONG
  CP=0, Board=Q: K(hi)=+2.73, Q(PAIR)=+1.06, J(hi)=-2.11  WRONG
  CP=0, Board=K: K(PAIR)=+3.26, Q(hi)=+1.22, J(hi)=-2.30  CORRECT
  CP=1, Board=J: K(hi)=+1.69, Q(hi)=-0.36, J(PAIR)=-0.88  WRONG
  CP=1, Board=Q: K(hi)=+2.45, Q(PAIR)=+1.24, J(hi)=-1.43  WRONG
  CP=1, Board=K: K(PAIR)=+2.95, Q(hi)=+1.29, J(hi)=-1.73  CORRECT

--- Player Symmetry Check ---
V(s, cp=0) should relate consistently to V(s, cp=1)

  J, Board=None, Pot=[1, 1]: V(cp=0)=-1.93, V(cp=1)=-1.33, diff=-0.60
  J, Board=None, Pot=[3, 3]: V(cp=0)=-1.94, V(cp=1)=-1.33, diff=-0.61
  J, Board=J, Pot=[1, 1]: V(cp=0)=-1.68, V(cp=1)=-0.86, diff=-0.82
  J, Board=J, Pot=[3, 3]: V(cp=0)=-1.67, V(cp=1)=-0.88, diff=-0.79
  J, Board=Q, Pot=[1, 1]: V(cp=0)=-2.09, V(cp=1)=-1.42, diff=-0.67
  J, Board=Q, Pot=[3, 3]: V(cp=0)=-2.11, V(cp=1)=-1.43, diff=-0.68
  J, Board=K, Pot=[1, 1]: V(cp=0)=-2.28, V(cp=1)=-1.71, diff=-0.57
  J, Board=K, Pot=[3, 3]: V(cp=0)=-2.30, V(cp=1)=-1.73, diff=-0.57
  Q, Board=None, Pot=[1, 1]: V(cp=0)=-0.86, V(cp=1)=-0.48, diff=-0.38
  Q, Board=None, Pot=[3, 3]: V(cp=0)=-0.77, V(cp=1)=-0.39, diff=-0.38
  Q, Board=J, Pot=[1, 1]: V(cp=0)=-0.29, V(cp=1)=-0.39, diff=+0.09
  Q, Board=J, Pot=[3, 3]: V(cp=0)=-0.24, V(cp=1)=-0.36, diff=+0.11
  Q, Board=Q, Pot=[1, 1]: V(cp=0)=+1.00, V(cp=1)=+1.20, diff=-0.20
  Q, Board=Q, Pot=[3, 3]: V(cp=0)=+1.06, V(cp=1)=+1.24, diff=-0.18
  Q, Board=K, Pot=[1, 1]: V(cp=0)=+1.14, V(cp=1)=+1.21, diff=-0.07
  Q, Board=K, Pot=[3, 3]: V(cp=0)=+1.22, V(cp=1)=+1.29, diff=-0.07
  K, Board=None, Pot=[1, 1]: V(cp=0)=+0.62, V(cp=1)=+0.63, diff=-0.01
  K, Board=None, Pot=[3, 3]: V(cp=0)=+0.79, V(cp=1)=+0.75, diff=+0.04
  K, Board=J, Pot=[1, 1]: V(cp=0)=+2.28, V(cp=1)=+1.60, diff=+0.68
  K, Board=J, Pot=[3, 3]: V(cp=0)=+2.36, V(cp=1)=+1.69, diff=+0.67
  K, Board=Q, Pot=[1, 1]: V(cp=0)=+2.56, V(cp=1)=+2.37, diff=+0.19
  K, Board=Q, Pot=[3, 3]: V(cp=0)=+2.73, V(cp=1)=+2.45, diff=+0.28
  K, Board=K, Pot=[1, 1]: V(cp=0)=+3.10, V(cp=1)=+2.85, diff=+0.25
  K, Board=K, Pot=[3, 3]: V(cp=0)=+3.26, V(cp=1)=+2.95, diff=+0.31

  Symmetry diff stats: mean=-0.165, std=0.433
  (Ideal: near zero mean if game is symmetric)

--- Full Prediction Range ---
  Predictions across 120 state encodings:
  Min: -2.378, Max: +3.572, Mean: +0.309, Std: 1.749
  (Reward range is [-13, +13]. Predictions should span a meaningful range.)

======================================================================
SECTION D: MC vs TD(0) TARGET VARIANCE COMPARISON
======================================================================

MC weighted avg per-state variance:    31.57
TD(0) weighted avg per-state variance: 5.64
Variance reduction ratio:              +82.1%

Breakdown by round:
  Round 0: MC var=36.60, TD var=1.48, reduction=+96.0%
  Round 1: MC var=25.00, TD var=11.08, reduction=+55.7%

Breakdown by step type:
      Terminal: MC var=14.26, TD var=14.26, reduction=+0.0%
  Non-terminal: MC var=36.27, TD var=1.93, reduction=+94.7%

======================================================================
SECTION E: NORMALIZATION METHODS ANALYSIS
======================================================================

Raw reward stats: mean=+0.236, std=6.810
Current MSE loss scale: ~31.6

Method 1: Divide by max reward (r/13)
  Normalized range:         [-1.000, +1.000]
  Global variance:          0.2745
  Weighted per-state var:   0.1868
  Expected MSE loss:        ~0.19
  Scale factor:             1/13^2 = 0.00592
  Effect: Purely cosmetic — loss_new = loss_old * 0.00592
          Network predictions would be in [-1, +1] range.

Method 2: Per-pot normalization (r / total_pot)
  a) r / player_pot:  range=[-13.00, +13.00], weighted var=19.7196
  b) r / total_pot:   range=[-6.50, +6.50], weighted var=3.5241
  Effect: Normalizes stakes so small-pot and big-pot games weigh equally.
          Within a state key, pot is constant, so this is per-state scaling.

Method 3: Running mean/std normalization ((r - mu) / sigma)
  mu = +0.236, sigma = 6.810
  Normalized range:         [-1.943, +1.874]
  Global variance:          1.0000
  Weighted per-state var:   0.6807
  Expected MSE loss:        ~0.68
  Effect: Centers rewards at 0, unit variance. Standard in PPO.
          Requires tracking running stats. Loss becomes ~0.68.

Method 4: Huber loss (delta=5) instead of MSE
  If predictions = per-state optimal (mean):
    MSE loss:   31.57
    Huber loss: 12.50
    Reduction:  60.4%
  Effect: Clips gradient for |error| > 5.
          Reduces sensitivity to outlier episodes (big pots).
          Does NOT reduce variance, but reduces its impact on training.
  35.4% of samples have |error from state mean| > 5
  (These are the outliers Huber would clip.)

======================================================================
SUMMARY & RECOMMENDATIONS
======================================================================

Key Findings:
  1. Weighted per-state MC variance:  31.57
     Observed training loss:          ~40
     -> The loss IS the irreducible MC variance floor.

  2. TD(0) variance:                  5.64
     Variance reduction:              +82.1%
     -> TD(0) would significantly reduce target variance.

Recommendations (in priority order):

  1. SWITCH TO TD(0) OR TD(lambda) TARGETS
     - Current MC targets assign the same episode-terminal reward to ALL
       decision points, even early ones with high outcome uncertainty.
     - TD(0) bootstraps from the value network, reducing variance at the
       cost of some bias (acceptable once the network is partially trained).
     - Expected loss reduction: ~82%

  2. NORMALIZE REWARDS
     - Divide by 13 (max reward) for clean [-1, +1] targets.
     - This is cosmetic for MSE but helps with learning rate tuning and
       prevents exploding gradients.
     - New expected loss: ~0.187 (much easier to interpret)

  3. CONSIDER HUBER LOSS
     - Replace MSE with Huber(delta=5) to reduce outlier sensitivity.
     - Won't reduce the loss floor but stabilizes training.

  4. IF LOSS DOESN'T MATCH VARIANCE (convergence issue):
     - Check learning rate (may need warmup or decay)
     - Check for gradient issues (exploding/vanishing)
     - Add gradient clipping
     - Verify batch size is large enough for stable gradients


======================================================================
REPORT COMPLETE
======================================================================
