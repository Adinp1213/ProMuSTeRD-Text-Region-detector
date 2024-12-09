import cv2
import numpy as np
import os
import glob

def load_ground_truth_boxes(gt_file):
    boxes = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split(',')))
            if len(parts) == 8:
            
                boxes.append(np.array(parts).reshape(-1, 2))
    return boxes

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def process_images(image_folder, gt_folder, scale, output_non_text, output_text,language):
    os.makedirs(output_non_text, exist_ok=True)
    os.makedirs(output_text, exist_ok=True)
    
    # Get all image and ground truth files
    image_files = sorted(glob.glob(os.path.join(image_folder, 'IMG_*.png')))
    gt_files = sorted(glob.glob(os.path.join(gt_folder, 'GT*.txt')))
    
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
                center_x = x + scale_w // 2
                center_y = y + scale_h // 2

                block_polygon = np.array([
                    [x, y],
                    [x + scale_w, y],
                    [x + scale_w, y + scale_h],
                    [x, y + scale_h]
                ], dtype=np.int32)

                is_text_block = any(is_point_in_polygon((center_x, center_y), box) for box in boxes)

                if is_text_block:
                    row_contains_text = True
                    filename = os.path.join(output_text, f"{os.path.basename(img_file).split('.')[0]}_text_block_mixed_{language}_{text_count}.png")
                    cv2.imwrite(filename, block)
                    text_count += 1
                    print('file added in text')
                    x += 20  
                else:
                    filename = os.path.join(output_non_text, f"{os.path.basename(img_file).split('.')[0]}_non_text_block_mixed_{language}_{non_text_count}.png")
                    cv2.imwrite(filename, block)
                    print('file added in non text')
                    non_text_count += 1
                    x += scale_w  

            y += 20 if row_contains_text else scale_h

process_images('dataset/AU-MSTR extended/Training/Devanagari', 'dataset/AU-MSTR extended/Training/Devanagari/GT', (50, 50), 'scale_50_non_text', 'scale_50_text','devanagari')
process_images('dataset/AU-MSTR extended/Training/Latin', 'dataset/AU-MSTR extended/Training/Latin/GT', (50, 50), 'scale_50_non_text', 'scale_50_text','Latin')
process_images('dataset/AU-MSTR extended/Training/Bengali', 'dataset/AU-MSTR extended/Training/Bengali/GT', (50, 50), 'scale_50_non_text', 'scale_50_text','bengali')
