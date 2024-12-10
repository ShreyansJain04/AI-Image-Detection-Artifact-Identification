import os
import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path
import argparse

def blur_image(image):
    """Apply Gaussian blur to the image."""
    kernel_size = random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def compress_image(image, quality=50):
    """Compress the image by saving it with lower quality."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    temp_file = "temp_compressed.jpg"
    pil_image.save(temp_file, 'JPEG', quality=quality)
    compressed_image = cv2.imread(temp_file)
    os.remove(temp_file)  # Clean up the temporary file
    return compressed_image

def process_image(image, blur_probability=0.5, compress_probability=0.5):
    """Process the image by applying blur and/or compression randomly."""
    processed_image = image.copy()

    # Apply random blur
    if random.random() < blur_probability:
        processed_image = blur_image(processed_image)
        print("Applied Blur to image.")
    
    # Apply random compression
    if random.random() < compress_probability:
        processed_image = compress_image(processed_image, quality=random.randint(30, 70))
        print("Applied Compression to image.")
    
    return processed_image

def augment_and_save(image_path, output_dir, input_dir, blur_probability=0.5, compress_probability=0.5):
    """Augment an image and save it to the output directory."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}. Skipping.")
        return

    # Generate the output subdirectory path
    relative_path = os.path.relpath(image_path, start=input_dir)
    output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))

    # Create subdirectory if it doesn't exist
    os.makedirs(output_subdir, exist_ok=True)

    # Original image name and path
    base_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_subdir, base_name)

    # Save the original image first
    cv2.imwrite(output_image_path, image)

    # Create and save two augmented versions (making 3 images in total)
    for i in range(2):
        augmented_image = process_image(image, blur_probability, compress_probability)
        augmented_image_path = os.path.join(output_subdir, f"{base_name.split('.')[0]}_augmented_{i+1}.jpg")
        cv2.imwrite(augmented_image_path, augmented_image)
        print(f"Saved augmented image: {augmented_image_path}")

def augment_images_in_directory(input_dir, output_dir, blur_probability=0.5, compress_probability=0.5):
    """Process all images in the directory and create augmented images."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Process image files only
                image_path = os.path.join(root, file)
                augment_and_save(image_path, output_dir, input_dir, blur_probability, compress_probability)

if __name__ == "__main__":
    # Specify the input directory (where original images are stored) and output directory
	parser = argparse.ArgumentParser(description="Augment images by applying random blur and compression.")
	parser.add_argument("input_dir", type=str, help="Directory containing original images")
	parser.add_argument("output_dir", type=str, help="Output directory for augmented images")
	parser.add_argument("--blur_probability", type=float, default=0.7, help="Probability of applying blur to an image")
	parser.add_argument("--compress_probability", type=float, default=0.5, help="Probability of applying compression to an image")

	args = parser.parse_args()

	# Run the augmentation process
	augment_images_in_directory(args.input_dir, args.output_dir, blur_probability=args.blur_probability, compress_probability=args.compress_probability)
	print("Augmentation process completed.")
