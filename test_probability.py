import tensorflow as tf
import cv2
import numpy as np

# Load the saved models and their corresponding scale sizes
model_paths = [
    'sobel_models/VGG_40x40_Sobel.h5',
    'sobel_models/VGG_50x50_Sobel.h5',
    'sobel_models/VGG_60x60_Sobel.h5',
    'sobel_models/VGG_70x70_Sobel.h5',
    'sobel_models/VGG_75x55_Sobel.h5',
    'sobel_models/VGG_80x80_Sobel.h5',
    'sobel_models/VGG_90x90_Sobel.h5'
]

scale_sizes = [(40, 40), (50, 50), (60, 60), (70, 70), (55, 75), (80, 80), (90, 90)]

models = [tf.keras.models.load_model(path) for path in model_paths]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image

def load_ground_truth_boxes(gt_file):
    boxes = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split(',')))
            if len(parts) == 8:
                boxes.append(np.array(parts).reshape(-1, 2))
    return boxes


def calculate_iou(image1, image2):
    intersection = cv2.bitwise_and(image1, image2)
    union = cv2.bitwise_or(image1, image2)
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    iou = intersection_count / union_count if union_count != 0 else 0
    return iou

def calculate_pixel_metrics(gt_image, result_image):
    assert gt_image.shape == result_image.shape, "Images must have the same dimensions"
    gt_pixels = gt_image.flatten()
    result_pixels = result_image.flatten()
    TP = np.sum((gt_pixels == 255) & (result_pixels == 255))
    FP = np.sum((gt_pixels == 0) & (result_pixels == 255))
    FN = np.sum((gt_pixels == 255) & (result_pixels == 0))
    TN = np.sum((gt_pixels == 0) & (result_pixels == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print metrics for debugging
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision: {precision}, Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    
    return f1_score,precision,recall

import os
import pandas as pd

def draw_ground_truth_boxes(image, boxes):
    mask = np.zeros_like(image)
    for box in boxes:
        box = box.astype(int)
        cv2.fillPoly(mask, [box], color=(255, 255, 255))
    return mask

def predict_image_scales(image, model, scale_size):
    height, width, _ = image.shape
    scale_w, scale_h = scale_size
    max_resolution = 1500
    if width > max_resolution or height > max_resolution:
        print(f"Skipping image: Resolution exceeds {max_resolution}px (width={width}, height={height})")
        return None  # Skip processing and return None or an appropriate value
    

    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height, scale_h):
        for x in range(0, width, scale_w):
            scale = image[y:y+scale_h, x:x+scale_w]

            if scale.shape[0] != scale_h or scale.shape[1] != scale_w:
                continue

            scale_input = np.expand_dims(scale, axis=0)
            scale_input = scale_input.astype('float32') / 255.0

            prediction = model.predict(scale_input)
            probability = prediction[0][0]

            intensity = int(probability * 255)
            output_image[y:y+scale_h, x:x+scale_w] = [intensity, intensity, intensity]

    return output_image

def apply_threshold_and_morphology(image, threshold):
    scaled_threshold = int(threshold * 255)
    _, thresholded = cv2.threshold(image, scaled_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((70, 70), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    return morphed

def apply_heatmap(image):
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)

def superimpose_images(base_image, overlay_image, alpha=0.6):
    return cv2.addWeighted(base_image, alpha, overlay_image, 1 - alpha, 0)

def save_and_show_output_image(output_image, output_path):
    if output_image.dtype != np.uint8:
        output_image = (output_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_image)


def update_or_append_excel(image_name, metrics):
    # Create DataFrame with new metrics
    new_df = pd.DataFrame([metrics], columns=['Image_Name', 'Threshold', 'IoU', 'Precision', 'Recall', 'F1_Score'])
    file_path = 'metrics_results_prob.xlsx'

    # Check if the Excel file already exists
    if os.path.exists(file_path):
        # If the file exists, read the existing data
        existing_df = pd.read_excel(file_path)
        # Append the new DataFrame to the existing data
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame
        existing_df = new_df
    
    # Save the combined DataFrame back to Excel
    existing_df.to_excel(file_path, index=False)

def main_process_2(id_num):
    input_image_path = f'dataset/AU-MSTR extended/Testing/Mixed/img_00{id_num}.png'
    gt_file_path = f'dataset/AU-MSTR extended/Testing/Mixed/GT/GT00{id_num}.txt'
    image = preprocess_image(input_image_path)
    # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    # sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    # image = np.uint8(sobel_combined)
    output_images = []
    for model, scale_size in zip(models, scale_sizes):
        output_image = predict_image_scales(image, model, scale_size)
        if output_image is not None: 
            output_images.append(output_image)
            #save_and_show_output_image(output_image, f'Output_{id_num}_{scale_size}.png')
        print(f"done for {scale_size}")

    if output_images:
        prob_sum = sum(img.astype('float32') for img in output_images)
        prob_avg = prob_sum / len(output_images)
        result_image = prob_avg.astype(np.uint8)
        #save_and_show_output_image(result_image, f'Output_result.png')
        threshold = 0.5
        thresholded_image = apply_threshold_and_morphology(result_image, threshold)
        heatmap = apply_heatmap(result_image)
        superimposed_image = superimpose_images(image, heatmap)
        save_and_show_output_image(superimposed_image, f'Superimposed_results_probability\Result_Mixed_00{id_num}.png')
    else:
        print('skipped')


def apply_bounding_box(image, original_image, classification_model):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold to ensure the image is binary
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if there are contours detected
    if not contours:
        print("No contours found.")
        return original_image

    # Find the contour with the largest area
    #largest_contour = max(contours, key=cv2.contourArea)
    for contour in contours:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contour)
        cropped_region = original_image[y:y + h, x:x + w]
        
        # Resize the cropped region to the input size of your classification model if needed
        cropped_resized = cv2.resize(cropped_region, (224, 224))  # Replace with your model's input size
        cropped_resized = np.expand_dims(cropped_resized, axis=0).astype('float32') / 255.0

        # Predict the script type using your trained classification model
        prediction = classification_model.predict(cropped_resized)
        script_label = np.argmax(prediction)  # Replace with your model's prediction logic

        # Set the bounding box color based on the classification
        if script_label == 0:  # Assuming 0 is Devanagari
            color = (0, 0, 255)  # Red for Devanagari
            script_name = "Devanagari"
        elif script_label == 1:  # Assuming 1 is Latin
            color = (255, 0, 0)  # Blue for Latin
            script_name = "Latin"
        elif script_label == 2:  # Assuming 2 is Bengali
            color = (0, 255, 255)  # Yellow for Bengali
            script_name = "Bengali"
        
        # Draw the bounding box with the appropriate color
        cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)
        
        # Annotate the detected script label near the bounding box
        cv2.putText(original_image, script_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return original_image


def save_metrics_to_excel(image_name, metrics, file_name='results.xlsx'):
    # Check if file exists
    file_exists = os.path.exists(file_name)
    
    # Create a DataFrame for the metrics
    df = pd.DataFrame([metrics], columns=['Image_Name','Threshold', 'IoU', 'Precision', 'Recall', 'F1_Score'])
    
    # Save to Excel (append if file exists, create new if not)
    if file_exists:
        with pd.ExcelWriter(file_name, mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False)
    else:
        with pd.ExcelWriter(file_name, mode='w') as writer:
            df.to_excel(writer, index=False)

def main_process(id_num):
    input_image_path = f'Total-Text\Test\img292.jpg'
    #gt_file_path = f'dataset/AU-MSTR extended/Testing/Devanagari/GT/GT0{id_num}.txt'
    image = preprocess_image(input_image_path)
    # sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    # sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    # image = np.uint8(sobel_combined)
    output_images = []
    for model, scale_size in zip(models, scale_sizes):
        output_image = predict_image_scales(image, model, scale_size)
        
        if output_image is not None: 
            output_images.append(output_image)
            output_image_path = f'output_image_{scale_size[0]}x{scale_size[1]}.png'
            save_and_show_output_image(output_image, output_image_path)
        print(f"done for {scale_size}")
    
    #classification_model = tf.keras.models.load_model('Multiclass_model\keras_model.h5')
    #ground_truth_boxes = load_ground_truth_boxes(gt_file_path)
    #gt_image = draw_ground_truth_boxes(image, np.array(ground_truth_boxes))
    #save_and_show_output_image(gt_image,'gmt_image.png')

    if output_images:
        prob_sum = sum(img.astype('float32') for img in output_images)
        prob_avg = prob_sum / len(output_images)
        result_image = prob_avg.astype(np.uint8)
        save_and_show_output_image(result_image,f'resultant_image_to_te_80_{id_num}.png')
        threshold = 0.8
        thresholded_image = apply_threshold_and_morphology(result_image, threshold)
        # f1, precision, recall = calculate_pixel_metrics(thresholded_image, gt_image)
        # iou = calculate_iou(thresholded_image, gt_image)
        # image_name = f'Latin_00{id_num}'
        # metrics = [image_name, threshold,iou, precision, recall, f1]
        # update_or_append_excel(image_name, metrics)
        save_and_show_output_image(thresholded_image, f'Result_threshold_to_te_80_{id_num}.png')
        heatmap = apply_heatmap(result_image)
        superimposed_image = superimpose_images(image, heatmap)
        save_and_show_output_image(superimposed_image, f'Result_to_te_80_{id_num}.png')
    else:
        print('skipped')

main_process(292)

#main_process_2(11)
# for i in range(55,10,1):
#     main_process_2(i)
#     print(f"Done for image 00{i}")

# for i in range(10,48,1):
#     main_process(i)
#     print(f"Done for image 0{i}")


# input_image_path = f'dataset/AU-MSTR extended/Testing/Devanagari/img_{id}.png'
# gt_file_path = f'dataset/AU-MSTR extended/Testing/Devanagari/GT/GT{id}.txt'

# # Load and process image
# image = preprocess_image(input_image_path)

# # Predict and create output images for each scale
# output_images = []
# for model, scale_size in zip(models, scale_sizes):
#     output_image = predict_image_scales(image, model, scale_size)
#     output_images.append(output_image)
#     output_image_path = f'output_image_{scale_size[0]}x{scale_size[1]}.png'
#     save_and_show_output_image(output_image, output_image_path)


# # Calculate the sum of probabilities for each pixel
# prob_sum = sum(img.astype('float32') for img in output_images)
# prob_avg = prob_sum / len(output_images)

# # Assign color based on average probability
# result_image = prob_avg.astype(np.uint8)

# # Save the average probability image
# cv2.imwrite('common_white.png', result_image)

# # Apply heatmap to the result image
# heatmap = apply_heatmap(result_image)
# cv2.imwrite('heatmap.png', heatmap)

# # Superimpose the heatmap on the original image
# superimposed_image = superimpose_images(image, heatmap)
# cv2.imwrite('superimposed_image.png', superimposed_image)
# save_and_show_output_image(superimposed_image, f'Superimposed_results_probability\Result_Deva_{id}.png')

