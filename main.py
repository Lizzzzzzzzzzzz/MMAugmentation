
import cv2
import numpy as np
import os
import random

def crop_center(img, crop_size):
    """
    Crop the center portion of an image to the specified size.
    
    Args:
        img: Input image
        crop_size: Size of the square crop
        
    Returns:
        Cropped image
    """
    height, width = img.shape[:2]
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    return img[start_y:start_y+crop_size, start_x:start_x+crop_size]

def calculate_black_ratio(img):
    """
    Calculate the proportion of black pixels in the image.
    
    Args:
        img: Binary image (255 for white, 0 for black)
        
    Returns:
        Ratio of black pixels to total pixels
    """
    return 1 - np.sum(img == 255) / img.size

def calculate_black_pixels(img):
    """
    Count the number of black pixels in the image.
    
    Args:
        img: Binary image (255 for white, 0 for black)
        
    Returns:
        Number of black pixels
    """
    return np.sum(img != 255)

def generate_target_ratios(original_ratio, num_samples=500):
    """
    Generate random target ratios normally distributed around the original ratio.
    
    Args:
        original_ratio: Starting ratio to center the distribution
        num_samples: Number of samples to generate
        
    Returns:
        Array of target ratios
    """
    # Standard deviation is 10% of the original ratio
    std_dev = original_ratio * 0.1
    target_ratios = np.random.normal(original_ratio, std_dev, num_samples)
    # Ensure values stay within reasonable bounds
    target_ratios = np.clip(target_ratios, 0.1, 0.9)
    return target_ratios

def get_multiple_random_positions(img, target_type, num_points):
    """
    Get multiple random pixel positions of the specified type.
    
    Args:
        img: Binary image
        target_type: 'black' or 'white', the type of pixels to select
        num_points: Number of points to return
        
    Returns:
        List of (y, x) coordinate tuples
    """
    height, width = img.shape
    target_value = 0 if target_type == 'black' else 255
    positions = np.where(img == target_value)
    
    if len(positions[0]) == 0:
        return []
    
    # Ensure we don't request more points than available
    num_points = min(num_points, len(positions[0]))
    indices = random.sample(range(len(positions[0])), num_points)
    return [(positions[0][i], positions[1][i]) for i in indices]

def apply_local_morphology(img, center_position, operation, kernel_size=3):
    """
    Apply a morphological operation locally around a specific position.
    
    Args:
        img: Input binary image
        center_position: (y, x) coordinate for the center of operation
        operation: Type of morphology ('erosion', 'dilation', 'opening', 'closing')
        kernel_size: Size of the square kernel for the operation
        
    Returns:
        Processed image with local morphology applied
    """
    height, width = img.shape
    y, x = center_position
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Ensure region stays within image boundaries
    y_start = max(0, y - kernel_size//2)
    y_end = min(height, y + kernel_size//2 + 1)
    x_start = max(0, x - kernel_size//2)
    x_end = min(width, x + kernel_size//2 + 1)
    
    temp_img = img.copy()
    region = temp_img[y_start:y_end, x_start:x_end]
    
    # Apply appropriate morphological operation
    if operation == 'erosion':
        processed_region = cv2.erode(region, kernel[:y_end-y_start, :x_end-x_start])
    elif operation == 'dilation':
        processed_region = cv2.dilate(region, kernel[:y_end-y_start, :x_end-x_start])
    elif operation == 'opening':
        processed_region = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel[:y_end-y_start, :x_end-x_start])
    else:  # closing
        processed_region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel[:y_end-y_start, :x_end-x_start])
    
    # Replace the region in the original image
    temp_img[y_start:y_end, x_start:x_end] = processed_region
    return temp_img

def process_image(img, target_ratio):
    """
    Process an image to achieve a target black/white ratio using improved strategies.
    
    Args:
        img: Input grayscale image
        target_ratio: Target ratio of black pixels to total pixels
        
    Returns:
        Tuple of (processed image, final ratio, list of operations used)
    """
    # Convert to binary image
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    # Calculate target number of black pixels
    total_pixels = binary.size
    target_black_pixels = int(target_ratio * total_pixels)
    
    # Initialize output image
    processed = binary.copy()
    operations_used = []
    max_iterations = 300
    
    # Parameters for adaptive processing
    min_points = 2
    max_points = 15
    
    for iteration in range(max_iterations):
        # Check current state
        current_black_pixels = calculate_black_pixels(processed)
        pixel_diff = abs(current_black_pixels - target_black_pixels)
        
        # Dynamically adjust parameters based on how far we are from the target
        if pixel_diff > 1000:
            tolerance = 10
            points_range = (8, max_points)
            kernel_size = 5
        elif pixel_diff > 500:
            tolerance = 8
            points_range = (5, 12)
            kernel_size = 3
        elif pixel_diff > 100:
            tolerance = 5
            points_range = (3, 8)
            kernel_size = 3
        else:
            tolerance = 3
            points_range = (min_points, 5)
            kernel_size = 3
        
        # If we're close enough to the target, stop processing
        if pixel_diff <= tolerance:
            break
        
        # Determine number of points to modify in this iteration
        points_per_iteration = random.randint(*points_range)
        
        # Choose operation type based on current state vs target
        if current_black_pixels < target_black_pixels:
            # Need more black pixels - use erosion or closing
            pixel_positions = get_multiple_random_positions(processed, 'black', points_per_iteration)
            if pixel_diff > 500:
                # Far from target: prefer erosion (7:3 ratio)
                operation = random.choice(['erosion'] * 7 + ['closing'] * 3)
            else:
                # Close to target: use more balanced approach
                operation = random.choice(['erosion'] * 5 + ['closing'] * 5)
        else:
            # Need fewer black pixels - use dilation or opening
            pixel_positions = get_multiple_random_positions(processed, 'white', points_per_iteration)
            if pixel_diff > 500:
                # Far from target: prefer dilation (7:3 ratio)
                operation = random.choice(['dilation'] * 7 + ['opening'] * 3)
            else:
                # Close to target: use more balanced approach
                operation = random.choice(['dilation'] * 5 + ['opening'] * 5)
        
        # If no suitable positions found, exit loop
        if not pixel_positions:
            break
        
        # Decide whether to use same operation for all points (70% chance)
        use_same_operation = random.random() < 0.7
        
        # Apply morphological operations to selected points
        temp_img = processed.copy()
        for pos in pixel_positions:
            if not use_same_operation:
                # Randomize operation for each point if not using same operation
                if current_black_pixels < target_black_pixels:
                    operation = random.choice(['erosion', 'closing'])
                else:
                    operation = random.choice(['dilation', 'opening'])
            temp_img = apply_local_morphology(temp_img, pos, operation, kernel_size)
        
        # Check if new image is closer to target
        new_black_pixels = calculate_black_pixels(temp_img)
        new_diff = abs(new_black_pixels - target_black_pixels)
        
        # Accept new image if it's better or slightly worse but random chance allows it
        if new_diff < pixel_diff or (new_diff < pixel_diff * 1.1 and random.random() < 0.3):
            processed = temp_img
            operations_used.append(operation)
    
    # Calculate final black ratio
    current_ratio = calculate_black_ratio(processed)
    return processed, current_ratio, operations_used

# Main program
input_folder = 'Your_Input_Folder'  # Replace with your actual input folder path
output_folder = 'Your_Output_Folder'  # Replace with your actual output folder path

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Image parameters
input_size = 256  # Size of input images
crop_size = 230   # Size of final cropped images

# Process each of the input images
for i in range(1, 5):
    image_path = os.path.join(input_folder, f'GI{i}.PNG')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Skip if image loading failed
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    
    # Resize image if needed
    if image.shape[0] != input_size or image.shape[1] != input_size:
        image = cv2.resize(image, (input_size, input_size))
    
    # Calculate original ratio of black pixels
    original_ratio = calculate_black_ratio(image)
    print(f"\nProcessing GI{i}.PNG (Original ratio: {original_ratio:.3f})")
    
    # Generate variations of target ratios
    target_ratios = generate_target_ratios(original_ratio)
    
    # Create 500 variations for each input image
    for j in range(500):
        target_ratio = target_ratios[j]
        
        # Process the image to achieve target ratio
        processed, final_ratio, operations = process_image(image, target_ratio)
        
        # Crop the center of the processed image
        processed_cropped = crop_center(processed, crop_size)
        cropped_ratio = calculate_black_ratio(processed_cropped)
        
        # Save the output image
        output_filename = f'HNP{(i-1)*500+j+1}.PNG'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, processed_cropped)
        
        # Print processing details
        print(f"  {output_filename}:")
        print(f"    Target ratio: {target_ratio:.3f}")
        print(f"    Operations: {operations}")
        print(f"    Ratio before crop: {final_ratio:.3f}")
        print(f"    Ratio after crop: {cropped_ratio:.3f}")

print("\nFinished processing images")
