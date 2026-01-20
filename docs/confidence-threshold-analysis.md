# Confidence Threshold Analysis for Lawn Mowing Segmentation

## Overview

This document analyzes the optimal confidence thresholds for the YCOR 3-class segmentation model used in autonomous lawn mowing applications. The analysis is based on asymmetric error tolerance patterns where different classes have different priorities and error costs.

## Class Priority Hierarchy

### 1. Non-traversable (Obstacles) - HIGHEST PRIORITY
- **Class Index**: 2
- **Color**: Red
- **Purpose**: Identify obstacles and hazards
- **Error Tolerance**: Asymmetric - cannot tolerate false negatives

### 2. Cuttable (Grass) - MEDIUM PRIORITY  
- **Class Index**: 0
- **Color**: Green
- **Purpose**: Identify areas that need cutting
- **Error Tolerance**: Asymmetric - cannot tolerate false negatives

### 3. Traversable (Paths) - LOWEST PRIORITY
- **Class Index**: 1
- **Color**: Gray
- **Purpose**: Identify safe paths for navigation
- **Error Tolerance**: Can tolerate false negatives

## Asymmetric Error Tolerance Analysis

### Non-traversable Class
```
✅ TOLERABLE: Other classes → Non-traversable (False Positive)
❌ INTOLERABLE: Non-traversable → Other classes (False Negative)

Example:
- Safe area labeled as obstacle → Robot avoids it → No harm, just inefficient
- Rock labeled as grass → Robot hits rock → Dangerous!
```

### Cuttable Class
```
✅ TOLERABLE: Cuttable → Non-traversable (False Positive)
❌ INTOLERABLE: Cuttable → Traversable (False Negative)

Example:
- Grass labeled as obstacle → Robot avoids grass → No cutting, but no damage
- Grass labeled as path → Robot drives over grass without cutting → Mission failure
```

### Traversable Class
```
✅ TOLERABLE: Traversable → Non-traversable (False Positive)
❌ INTOLERABLE: Traversable → Cuttable (False Negative)

Example:
- Path labeled as obstacle → Robot avoids path → Takes longer route, but safe
- Path labeled as grass → Robot tries to cut path → Unnecessary work, but not dangerous
```

## Error Cost Matrix

| True Class | Predicted Class | Cost | Impact |
|------------|----------------|------|---------|
| Non-traversable | Cuttable | 10.0 | Hit obstacle thinking it's grass |
| Non-traversable | Traversable | 10.0 | Hit obstacle thinking it's path |
| Cuttable | Traversable | 5.0 | Miss grass that needs cutting |
| Cuttable | Non-traversable | 1.0 | Avoid grass (false obstacle) |
| Traversable | Cuttable | 2.0 | Cut path unnecessarily |
| Traversable | Non-traversable | 1.0 | Avoid path (false obstacle) |

## Recommended Confidence Thresholds

### Conservative Approach (Safety First)
```bash
--conf-threshold 0.4,0.6,0.8  # Non-traversable, Cuttable, Traversable
```

### Balanced Approach
```bash
--conf-threshold 0.5,0.7,0.8  # Non-traversable, Cuttable, Traversable
```

### Aggressive Approach (Higher Precision)
```bash
--conf-threshold 0.6,0.8,0.9  # Non-traversable, Cuttable, Traversable
```

## Threshold Selection Rationale

### Non-traversable: Lower Threshold (0.4-0.6)
- **Goal**: Maximize recall (catch all obstacles)
- **Strategy**: Lower threshold = higher recall = fewer missed obstacles
- **Priority**: Safety-critical

### Cuttable: Medium Threshold (0.6-0.8)
- **Goal**: Balance recall and precision
- **Strategy**: Medium threshold = good recall + reasonable precision
- **Priority**: Mission-critical

### Traversable: Higher Threshold (0.8-0.9)
- **Goal**: Maximize precision (avoid false paths)
- **Strategy**: Higher threshold = higher precision = fewer false positives
- **Priority**: Efficiency-focused

## Confidence Score Analysis Results

Based on analysis of the YCOR model output:

### Confidence Distribution
- **Range**: 0.4222 to 1.0000 (42.22% to 100% confidence)
- **Mean**: 0.9569 (95.69% average confidence)
- **Std**: 0.0875 (8.75% standard deviation)
- **Median**: 0.9923 (99.23% median confidence)

### Pixel Distribution by Confidence
- **< 0.5**: 334 pixels (0.1%) - Very uncertain
- **0.5-0.7**: 19,619 pixels (3.5%) - Low confidence
- **0.7-0.9**: 52,095 pixels (9.4%) - Medium confidence
- **≥ 0.9**: 485,008 pixels (87.1%) - High confidence

## Implementation Guidelines

### 1. Start with Conservative Thresholds
```bash
# Begin with safety-first approach
--conf-threshold 0.4,0.6,0.8
```

### 2. Monitor Performance Metrics
- **Non-traversable recall**: Track obstacle detection rate
- **Cuttable recall**: Track grass detection rate  
- **Traversable precision**: Track path detection accuracy

### 3. Adjust Based on Real-World Performance
- **Increase threshold** if too many false positives
- **Decrease threshold** if too many false negatives
- **Monitor collision incidents** and adjust accordingly

### 4. Environment-Specific Tuning
- **Daytime**: Can use higher thresholds (better visibility)
- **Evening/Low light**: Use lower thresholds (more conservative)
- **Challenging terrain**: Use lower thresholds (safety first)

## Validation Strategy

### Class-Specific Metrics
```python
# Focus on recall for safety-critical classes
non_traversable_recall = calculate_recall(non_traversable_predictions)
cuttable_recall = calculate_recall(cuttable_predictions)

# Focus on precision for efficiency classes
traversable_precision = calculate_precision(traversable_predictions)
```

### Weighted Evaluation
```python
# Weight metrics by class priority
weighted_score = (
    0.5 * non_traversable_recall +      # Highest weight (safety)
    0.3 * cuttable_recall +             # Medium weight (mission)
    0.2 * traversable_precision         # Lower weight (efficiency)
)
```

## Usage Examples

### Single Image Processing
```bash
# Conservative approach
python tools/inference.py \
    --input image.jpg \
    --config config.py \
    --checkpoint model.pth \
    --output-dir results/ \
    --conf-threshold 0.4,0.6,0.8

# Balanced approach
python tools/inference.py \
    --input image.jpg \
    --config config.py \
    --checkpoint model.pth \
    --output-dir results/ \
    --conf-threshold 0.5,0.7,0.8
```

### Video Processing
```bash
# With FPS overlay
python tools/inference.py \
    --input video.mp4 \
    --config config.py \
    --checkpoint model.pth \
    --output-dir results/ \
    --conf-threshold 0.5,0.7,0.8 \
    --overlay-fps
```

## Monitoring and Maintenance

### Key Performance Indicators (KPIs)
1. **Safety**: Zero collision incidents
2. **Mission Success**: >90% grass coverage
3. **Efficiency**: <20% false obstacle detections

### Regular Validation
- **Weekly**: Check confidence distributions on new data
- **Monthly**: Validate thresholds on diverse scenarios
- **Quarterly**: Full performance review and threshold adjustment

### Alert Conditions
- **Recall drop**: Non-traversable recall < 95%
- **Precision drop**: Traversable precision < 80%
- **Collision incidents**: Any obstacle-related collisions

## Conclusion

The confidence threshold selection should prioritize **safety over efficiency**, with the following hierarchy:

1. **Safety First**: Non-traversable class gets lowest threshold (highest recall)
2. **Mission Success**: Cuttable class gets medium threshold (balanced)
3. **Efficiency**: Traversable class gets highest threshold (highest precision)

This asymmetric approach ensures that the autonomous lawn mower operates safely while maintaining mission effectiveness.

