import cv2
import numpy as np
import os

"""
Convert image to grayscale, blurs it to remove noise, and apply adaptive thresholding to make the grid pop
"""
def preprocess_image(image):
    # Convert to Grayscale and Gaussian Blur to smooth out paper texture/noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Turn image to black and white and invert so the grid lines and numbers are white, and background black. 
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh

"""
Find largest square-like contour in the image (the sudoku board)
"""
def find_board_contour(thresh_image):

    # Find all contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # approx the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # if the polygon has 4 corners, we found the board
        if len(approx) == 4:
            return approx
            
    return None

"""
Take the 4 corners of the board and warp image so it becomes a flat top-down square
"""
def warp_perspective(image, contour):

    # reshape the contour points and order them: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-Left: has the smallest x+y sum
    rect[2] = pts[np.argmax(s)] # Bottom-Right: has the largest x+y sum
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-Right: has the smallest y-x difference
    rect[3] = pts[np.argmax(diff)] # Bottom-Left: has the largest y-x difference
    
    (tl, tr, br, bl) = rect
    
    # calculate the max width and height for the new perfect square
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    side = max(maxWidth, maxHeight)
    
    # destination points for the top-down view
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")
    
    # calculate the perspective transform matrix and warp it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (side, side))
    
    return warped

"""
Extract sudoku board from image
"""
def extract_board(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    thresh = preprocess_image(image)
    board_contour = find_board_contour(thresh)
    
    if board_contour is None:
        raise ValueError("Could not find Sudoku board in the image.")
        
    warped_board = warp_perspective(image, board_contour)

    # warp grayscale/thresholded image to feed clean numbers to CNN later
    warped_gray = warp_perspective(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), board_contour)
    
    return warped_board, warped_gray

"""
Slice 9x9 Sudoku grid into 81 individual cell images
"""
def split_boxes(warped_image):

    # make image dimensions divisible by 9
    side = warped_image.shape[0]
    side = side - (side % 9) 
    warped_image = cv2.resize(warped_image, (side, side))
    
    # split image vertically into 9 rows, then into 9 columns
    rows = np.vsplit(warped_image, 9)
    boxes = []
    
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            # crop a few pixels off all 4 sides of the box to remove thick grid lines
            box = box[4:-4, 4:-4] 
            box = cv2.resize(box, (28, 28))
            _, box = cv2.threshold(box, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            boxes.append(box)
            
    return boxes 


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(BASE_DIR, "test_images", "test2.png")

    try:
        color_board, gray_board = extract_board(image_path)

        # create output folder if it doesn't exist
        output_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        # save results
        color_path = os.path.join(output_dir, "warped_color.jpg")
        gray_path = os.path.join(output_dir, "warped_gray.jpg")

        cv2.imwrite(color_path, color_board)
        cv2.imwrite(gray_path, gray_board)

        boxes = split_boxes(gray_board)
        
        # create folder just for the slices
        slices_dir = os.path.join(output_dir, "slices")
        os.makedirs(slices_dir, exist_ok=True)
        
        for i, box in enumerate(boxes):
            box_path = os.path.join(slices_dir, f"box_{i}.jpg")
            cv2.imwrite(box_path, box)
            
        print(f"Saved 81 individual cells to: {slices_dir}")
        print(f"Saved color board to: {color_path}")
        print(f"Saved gray board to: {gray_path}")
    except Exception as e:
        print(f"Error: {e}")
