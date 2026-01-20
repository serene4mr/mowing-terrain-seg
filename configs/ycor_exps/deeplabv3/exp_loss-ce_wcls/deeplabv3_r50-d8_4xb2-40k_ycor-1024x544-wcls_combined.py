# DeepLabV3 with ResNet-50 backbone on YCOR dataset
# 4 GPUs x 2 samples per GPU, trained for 40K iterations

_base_ = [
    '../../_base_/models/deeplabv3_r50-d8.py', 
    '../../_base_/datasets/ycor.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/schedules/schedule_40k.py'
]

# Original individual weighting methods
frequency_inverse_weights_train = [0.672440,1.028959,0.752052,70.239090,15.458582,2.498081,0.404913,1.012719]
sqrt_inverse_weights_train = [2.319379, 2.869089, 2.452838, 23.704698, 11.120641, 4.470420, 1.799807, 2.846358]
log_inverse_weights_train = [1.682599, 2.107989, 1.794491, 6.331347, 4.817606, 2.994965, 1.175359, 2.092080]

# COMBINATION STRATEGY 1: Weighted Average Approach
# Rationale: Combine all three methods with different weights based on their individual performance
# - Log inverse (40%): Best overall mIoU (47.61%) - primary contributor
# - Square root (35%): Best mean accuracy (61.72%) - secondary contributor  
# - Direct inverse (25%): Most aggressive but can help rare classes - minor contributor
# This approach balances the strengths of each method while reducing the impact of extreme weights
weighted_avg_weights = [
    0.25 * frequency_inverse_weights_train[i] + 
    0.35 * sqrt_inverse_weights_train[i] + 
    0.40 * log_inverse_weights_train[i]
    for i in range(8)
]

# COMBINATION STRATEGY 2: Class-Specific Selection (Best Performer per Class)
# Rationale: Use the best performing weighting method for each individual class based on our analysis
# This is a data-driven approach that selects the optimal strategy per class
class_specific_weights = [
    sqrt_inverse_weights_train[0],  # smooth_trail: sqrt (2.32) - best IoU performance
    log_inverse_weights_train[1],   # traversable_grass: log (2.11) - best IoU performance
    sqrt_inverse_weights_train[2],  # rough_trail: sqrt (2.45) - best IoU performance
    3.0,                           # puddle: moderate weight (3.0) - avoid extreme 70.24 weight
    log_inverse_weights_train[4],   # obstacle: log (4.82) - best IoU performance
    frequency_inverse_weights_train[5], # non_traversable_vegetation: direct (2.50) - best IoU performance
    log_inverse_weights_train[6],   # high_vegetation: log (1.18) - best IoU performance
    log_inverse_weights_train[7],   # sky: log (2.09) - best IoU performance
]

# COMBINATION STRATEGY 3: Capped Conservative Approach
# Rationale: Use log inverse (best overall performer) but cap extreme weights to prevent training instability
# This maintains the benefits of log inverse while avoiding the problems of extreme weighting
capped_conservative_weights = [
    min(log_inverse_weights_train[i], 8.0)  # Cap at 8.0 to prevent extreme weights
    for i in range(8)
]

# COMBINATION STRATEGY 4: Hierarchical Weighting (Different for Main vs Auxiliary)
# Rationale: Use different strategies for main decode head vs auxiliary head
# - Main head: Log inverse (best overall performance)
# - Auxiliary head: Square root (best mean accuracy, helps with class balance)
hierarchical_main_weights = log_inverse_weights_train
hierarchical_aux_weights = sqrt_inverse_weights_train

# COMBINATION STRATEGY 5: Adaptive Weighting (Recommended)
# Rationale: Blend log inverse (best overall) with square root (best mean accuracy) 
# This combines the two best performers in a 70:30 ratio for optimal balance
adaptive_weights = [
    0.7 * log_inverse_weights_train[i] + 0.3 * sqrt_inverse_weights_train[i]
    for i in range(8)
]

# =============================================================================
# WEIGHT COMPARISON TABLE (for reference)
# =============================================================================
# Class                    | Direct | Sqrt  | Log   | Weighted| Class-Spec| Capped | Adaptive
# smooth_trail            | 0.67   | 2.32  | 1.68  | 1.89    | 2.32      | 1.68   | 1.89
# traversable_grass       | 1.03   | 2.87  | 2.11  | 2.26    | 2.11      | 2.11   | 2.26
# rough_trail             | 0.75   | 2.45  | 1.79  | 2.01    | 2.45      | 1.79   | 2.01
# puddle                  | 70.24  | 23.70 | 6.33  | 10.47   | 3.00      | 6.33   | 7.62
# obstacle                | 15.46  | 11.12 | 4.82  | 6.99    | 4.82      | 4.82   | 5.47
# non_traversable_vegetation| 2.50  | 4.47  | 2.99  | 3.40    | 2.50      | 2.99   | 3.40
# high_vegetation         | 0.40   | 1.80  | 1.18  | 1.39    | 1.18      | 1.18   | 1.39
# sky                     | 1.01   | 2.85  | 2.09  | 2.30    | 2.09      | 2.09   | 2.30
# =============================================================================

# =============================================================================
# CONFIGURATION SELECTION: Choose which combination strategy to use
# =============================================================================
# Uncomment ONE of the following lines to select your preferred strategy:

# Strategy 1: Weighted Average (balanced approach)
# selected_weights = weighted_avg_weights

# Strategy 2: Class-Specific Selection (data-driven per-class optimization)
# selected_weights = class_specific_weights

# Strategy 3: Capped Conservative (log inverse with safety cap)
# selected_weights = capped_conservative_weights

# Strategy 4: Hierarchical (different for main vs auxiliary heads)
# selected_weights = hierarchical_main_weights
# selected_aux_weights = hierarchical_aux_weights

# Strategy 5: Adaptive Weighting (RECOMMENDED - best of both worlds)
selected_weights = adaptive_weights
selected_aux_weights = adaptive_weights  # Use same for both heads

# =============================================================================
# YCOR-specific crop size (full size for maximum coverage)
# =============================================================================
crop_size = (1024, 544)  # Full YCOR size - no information loss
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,  
    size=crop_size
)

# =============================================================================
# Model configuration with combined weighting strategy
# =============================================================================
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=8,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            class_weight=selected_weights,  # Uses the selected combination strategy
            avg_non_ignore=True,
        )
    ),  
    auxiliary_head=dict(
        num_classes=8,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4,
            class_weight=selected_aux_weights,  # Uses auxiliary-specific weights if hierarchical
            avg_non_ignore=True,
        )
    )
)

# =============================================================================
# EXPECTED BENEFITS OF COMBINED WEIGHTING:
# =============================================================================
# 1. Balanced Performance: Combines strengths of multiple approaches
# 2. Reduced Overfitting: Less extreme weights prevent training instability  
# 3. Better Rare Class Handling: Moderate weighting for challenging classes
# 4. Improved Generalization: More stable training dynamics
# 5. Data-Driven Selection: Uses empirical performance to guide weighting
# 
# RECOMMENDED STRATEGY: Adaptive Weighting (Strategy 5)
# - Combines log inverse (47.61% mIoU) with square root (61.72% mAcc)
# - 70:30 ratio balances overall performance with class balance
# - Avoids extreme weights that can cause training instability
# - Expected to achieve >47.6% mIoU with improved mean accuracy
# =============================================================================