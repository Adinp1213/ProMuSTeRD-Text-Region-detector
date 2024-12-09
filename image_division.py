import cv2
import numpy as np
import os
import glob

def apply_morphological_operations(block):
    # Convert the image to grayscale
    gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, binary_block = cv2.threshold(gray_block, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Apply dilation followed by erosion (closing operation)
    kernel = np.ones((3, 3), np.uint8)
    morph_block = cv2.morphologyEx(binary_block, cv2.MORPH_CLOSE, kernel)
    # Convert back to 3-channel image
    morph_block_colored = cv2.cvtColor(morph_block, cv2.COLOR_GRAY2BGR)
    return morph_block_colored


def load_ground_truth_boxes(gt_file):
    boxes = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = list(map(int, line.split(',')))
            if len(parts) == 8:
                # Convert coordinates to a polygon (rectangle)
                boxes.append(np.array(parts).reshape(-1, 2))
    return boxes

def polygon_to_bounding_rect(polygon):
    x_coords, y_coords = zip(*polygon)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return (x_min, y_min, x_max, y_max)

def intersection_area(rect1, rect2):
    # Compute the intersection rectangle
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])
    
    # Calculate intersection area
    if x_right > x_left and y_bottom > y_top:
        return (x_right - x_left) * (y_bottom - y_top)
    return 0

def process_images(image_folder, gt_folder, scale, output_non_text, output_text, language):
    os.makedirs(output_non_text, exist_ok=True)
    os.makedirs(output_text, exist_ok=True)
    
    # Get all image and ground truth files
    image_files = sorted(glob.glob(os.path.join(image_folder, 'img_*.png')))
    gt_files = sorted(glob.glob(os.path.join(gt_folder, 'GT*.txt')))
    
    print(image_files)
    for img_file, gt_file in zip(image_files, gt_files):
        image = cv2.imread(img_file)
        height, width, _ = image.shape

        boxes = load_ground_truth_boxes(gt_file)
        scale_w, scale_h = scale
        non_text_count = 0
        text_count = 0

        y = 0
        while y + scale_h <= height:
            x = 0
            row_contains_text = False
            
            while x + scale_w <= width:
                block = image[y:y+scale_h, x:x+scale_w]
                block_rect = (x, y, x + scale_w, y + scale_h)
                
                total_intersection_area = 0
                for box in boxes:
                    box_rect = polygon_to_bounding_rect(box)
                    intersection = intersection_area(block_rect, box_rect)
                    total_intersection_area += intersection
                
                block_area = scale_w * scale_h
                intersection_ratio = total_intersection_area / block_area
                
                if intersection_ratio >= 0.90:
                    row_contains_text = True
                    filename = os.path.join(output_text, f"{os.path.basename(img_file).split('.')[0]}_latin_block_{language}_{text_count}.png")
                    cv2.imwrite(filename, block)
                    text_count += 1
                    print('file added in text')
                    x += 20  
                elif total_intersection_area > 0:

                    print('Intersection found but below 95% threshold')
                    x += scale_w  
                else:
                    filename = os.path.join(output_non_text, f"{os.path.basename(img_file).split('.')[0]}_non_text_block_{language}_{non_text_count}.png")
                    cv2.imwrite(filename, block)
                    print('file added in non text')
                    non_text_count += 1
                    x += scale_w  # Move x to the next block

            # Move y for the next row based on whether text was detected
            y += 20 if row_contains_text else scale_h

# Example usag7
# process_images('dataset/AU-MSTR extended/Training/Devanagari', 'dataset/AU-MSTR extended/Training/Devanagari/GT', (100, 100), 'scale_100_non_text', 'scale_100_text','devanagari')
# process_images('dataset/AU-MSTR extended/Training/Latin', 'dataset/AU-MSTR extended/Training/Latin/GT', (100, 100), 'scale_100_non_text', 'scale_100_text','Latin')
# process_images('dataset/AU-MSTR extended/Training/Bengali', 'dataset/AU-MSTR extended/Training/Bengali/GT', (100, 100), 'scale_100_non_text', 'scale_100_text','bengali')

#process_images('sobel_Bengali', 'dataset/AU-MSTR extended/Training/Bengali/GT', (40, 40), 'scale_sobel_40_non_text', 'scale_sobel_40_text','ben')
process_images('dataset\AU-MSTR extended\Testing\Latin', 'dataset/AU-MSTR extended/Testing/Latin/GT', (100, 100), 'scale_90_non_text', 'scale_90_text','ben')
#process_images('dataset\AU-MSTR extended\Testing\Bengali', 'dataset/AU-MSTR extended/Testing/Bengali/GT', (80, 80), 'scale_80_non_text', 'scale_80_ben','ben')
#process_images('dataset\AU-MSTR extended\Testing\Devanagari', 'dataset/AU-MSTR extended/Testing/Devanagari/GT', (80, 80), 'scale_80_non_text', 'scale_80_deva','deva')
#process_images('dataset\AU-MSTR extended\Training\Latin', 'dataset/AU-MSTR extended/Training/Latin/GT', (80, 80), 'scale_80_non_text', 'scale_80_latin','lat')
