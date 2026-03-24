import cv2
import numpy as np
import os
import random

"""Create folders for classes 0 through 9."""
def create_directory_structure(base_path):
    for i in range(10):
        os.makedirs(os.path.join(base_path, str(i)), exist_ok=True)

"""Generates an empty box (Class 0) with slight background noise/artifacts."""
def generate_empty_cell(size=28):
    img = np.zeros((size, size), dtype=np.uint8)
    
    # add random noise to simulate paper texture or imperfect thresholding
    noise = np.random.randint(0, 50, (size, size), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # randomly add faint white edge
    if random.random() > 0.5:
        thickness = random.randint(1, 2)
        cv2.rectangle(img, (0, 0), (size-1, size-1), 255, thickness)
        
    return img

"""Generates cell containing a specific digit with randomized fonts and sizes."""
def generate_digit_cell(digit, size=28):
    img = np.zeros((size, size), dtype=np.uint8)
    
    # randomize the look of the digit
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]
    font = random.choice(fonts)
    scale = random.uniform(0.6, 0.9)
    thickness = random.randint(1, 2)
    
    text_size = cv2.getTextSize(str(digit), font, scale, thickness)[0]
    
    # add random offset to numbers
    offset_x = random.randint(-2, 2)
    offset_y = random.randint(-2, 2)
    
    text_x = (size - text_size[0]) // 2 + offset_x
    text_y = (size + text_size[1]) // 2 + offset_y
    
    cv2.putText(img, str(digit), (text_x, text_y), font, scale, 255, thickness)
    
    # apply  slight blur randomly
    if random.random() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
    return img

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    create_directory_structure(dataset_dir)
    
    samples_per_class = 1000  # generate 1000 imgs
    print(f"Generating {samples_per_class * 10} images. Please wait...")
    
    for class_id in range(10):
        folder_path = os.path.join(dataset_dir, str(class_id))
        
        for i in range(samples_per_class):
            if class_id == 0:
                img = generate_empty_cell()
            else:
                img = generate_digit_cell(class_id)
                
            file_path = os.path.join(folder_path, f"{class_id}_{i}.jpg")
            cv2.imwrite(file_path, img)
            
    print(f"Dataset generation complete! Saved to {dataset_dir}")

if __name__ == "__main__":
    main()
