# PNG Format for Semantic Segmentation Datasets

## 🎯 Overview

PNG (Portable Network Graphics) files used in semantic segmentation datasets often employ **indexed color format** to efficiently store both visual information and semantic class IDs.

## 🎨 PNG File Structure

### A. Two Main Components
```python
png_file = {
    'pixel_data': [[8, 8, 8], [1, 2, 1], [0, 0, 0]],  # Class IDs
    'color_palette': {
        0: (255, 255, 255),  # ID 0 → White
        1: (178, 176, 153),  # ID 1 → Light Gray
        2: (128, 255, 0),    # ID 2 → Lime Green
        # ... more mappings
    }
}
```

### B. Indexed Color Format (Mode 'P')
- **Each pixel stores**: A palette index (0-255)
- **Color palette**: Maps each index to an RGB color
- **Storage efficiency**: 1 byte per pixel instead of 3 bytes (RGB)

## 🔍 How It Works

### A. Storage
```python
# Each pixel in the PNG stores:
pixel_value = 8  # "Use palette index 8"

# The palette says:
palette[8] = (1, 88, 255)  # "Index 8 = RGB(1, 88, 255) = blue"
```

### B. Display vs Loading
```python
# When you open the file (display):
# 1. Read pixel: "use palette index 8"
# 2. Look up palette: "index 8 = blue"
# 3. Show: blue color

# When we load with PIL (programmatic):
# 1. Read pixel: "use palette index 8"
# 2. Return: 8 (the index number)
# 3. Use: class ID 8
```

## 🚀 Why This is Perfect for Semantic Segmentation

### A. Dual Purpose
- ✅ **Human-friendly**: You see meaningful colors when opening the file
- ✅ **Machine-friendly**: We get class IDs directly without conversion
- ✅ **Efficient**: Compact storage format

### B. No Conversion Needed
```python
# ❌ We DON'T need this:
# rgb_color = get_pixel_color(mask, x, y)
# class_id = convert_color_to_class(rgb_color)

# ✅ We DO get this directly:
# class_id = mask_array[y, x]  # Already the class ID!
```

## 📊 Example: Yamaha Dataset

### A. Dataset Structure
```python
# Yamaha semantic segmentation dataset:
yamaha_palette = {
    0: (255, 255, 255),  # Background
    1: (178, 176, 153),  # Smooth trail
    2: (128, 255, 0),    # Traversable grass
    3: (156, 76, 30),    # Rough trail
    4: (255, 0, 128),    # Puddle
    5: (255, 0, 0),      # Obstacle
    6: (0, 160, 0),      # Non-traversable vegetation
    7: (40, 80, 0),      # High vegetation
    8: (1, 88, 255)      # Sky
}
```

### B. Loading Process
```python
def load_semantic_mask(mask_path):
    """Load semantic mask - PNG contains IDs + color mapping"""
    mask_image = Image.open(mask_path)  # Mode: 'P' (Palette)
    mask_array = np.array(mask_image)   # Get class IDs from pixels
    
    # mask_array contains: 0, 1, 2, 3, 4, 5, 6, 7, 8
    # These ARE the semantic class IDs!
    
    return mask_array
```

## 🔧 Technical Details

### A. PIL/Pillow Handling
```python
# PIL automatically handles indexed color PNGs:
mask_image = Image.open("labels.png")
print(mask_image.mode)  # 'P' (Palette)

# Converting to array gives palette indices:
mask_array = np.array(mask_image)
print(mask_array.dtype)  # uint8 (0-255)
```

### B. Palette Access
```python
# Access the color palette if needed:
palette = mask_image.palette
if hasattr(palette, 'colors'):
    for index, rgb_color in palette.colors.items():
        print(f"Index {index} → RGB{rgb_color}")
```

## 🎯 Key Benefits

### A. Storage Efficiency
- **Indexed color**: 1 byte per pixel
- **RGB format**: 3 bytes per pixel
- **Savings**: 66% storage reduction

### B. Semantic Clarity
- **Class IDs**: Directly accessible
- **Visual colors**: Meaningful for humans
- **No ambiguity**: Each ID maps to one class

### C. Processing Speed
- **No conversion**: Class IDs ready to use
- **Fast loading**: Efficient format
- **Memory efficient**: Smaller file sizes

## 📋 Common Use Cases

### A. Semantic Segmentation Datasets
- **COCO**: Uses indexed color PNGs
- **Pascal VOC**: Uses indexed color PNGs
- **Yamaha**: Uses indexed color PNGs
- **Cityscapes**: Uses indexed color PNGs

### B. Class Mapping
```python
# Original classes (9 classes)
original_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Custom mapping (4 classes)
custom_mapping = {
    0: 0,  # background → background
    1: 1,  # smooth trail → road
    2: 1,  # traversable grass → road
    3: 1,  # rough trail → road
    4: 2,  # puddle → water
    5: 3,  # obstacle → obstacle
    6: 3,  # vegetation → obstacle
    7: 3,  # high vegetation → obstacle
    8: 0,  # sky → background
}
```

## 🚨 Important Notes

### A. Mode Detection
```python
# Always check the image mode:
mask_image = Image.open("labels.png")
if mask_image.mode == 'P':
    # Indexed color format - perfect for semantic segmentation
    mask_array = np.array(mask_image)  # Get class IDs
elif mask_image.mode == 'RGB':
    # True color format - needs conversion
    # This is less common for semantic segmentation
```

### B. Class ID Range
```python
# Check the class ID range:
unique_classes = np.unique(mask_array)
print(f"Classes found: {unique_classes}")
print(f"Class range: {unique_classes.min()} - {unique_classes.max()}")
```

## 🎨 Summary

**PNG files for semantic segmentation:**

1. ✅ **Store pixel data as class IDs** (0, 1, 2, 3, 4, 5, 6, 7, 8)
2. ✅ **Include color palette** for visual display
3. ✅ **Efficient storage** (indexed color format)
4. ✅ **Direct class ID access** (no conversion needed)
5. ✅ **Human and machine friendly**

**This format is the standard for semantic segmentation datasets because it provides the perfect balance of efficiency, clarity, and usability.**
