import base64
import os
import copy
import cv2
import numpy as np
import torch

from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.cv_pipeline import preprocess_image, find_board_contour, warp_perspective, split_boxes
from models.cnn_build import SudokuCNN
from core.solver import solve_sudoku

def print_grid(grid, title="Sudoku Board"):
    print(f"\n--- {title} ---")
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        
        row_str = ""
        for j in range(9):
            # print vertical dividers for 3x3 boxes
            if j % 3 == 0 and j != 0:
                row_str += "| "
        
            val = grid[i][j]
            row_str += str(val) if val != 0 else "."
            row_str += " "
            
        print(row_str)
    print("-" * 25 + "\n")

app = FastAPI(title="Sudoku Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading PyTorch model on: {device}")

model = SudokuCNN().to(device)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, "models", "saved_weights", "sudoku_cnn.pth")

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
else:
    print("WARNING: Model weights not found! Ensure you have trained the model.")

# use same transformations during training 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.get("/")
def read_root():
    return {"message": "Sudoku Backend is operational!"}

@app.post("/solve")
async def solve_sudoku_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    thresh = preprocess_image(image)
    board_contour = find_board_contour(thresh)
    
    if board_contour is None:
        raise HTTPException(status_code=400, detail="Could not detect a Sudoku board in the image.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    warped_gray = warp_perspective(gray_image, board_contour)
    boxes = split_boxes(warped_gray) 

    # inference 
    tensor_boxes = []
    for box in boxes:
        # convert numpy array to PIL Image
        pil_img = Image.fromarray(box)
        tensor = transform(pil_img)
        tensor_boxes.append(tensor)
        
    # [81, 1, 28, 28]
    batch_tensors = torch.stack(tensor_boxes).to(device)
    
    with torch.no_grad():
        outputs = model(batch_tensors)
        _, predicted_classes = torch.max(outputs, 1)
        
    predictions = predicted_classes.cpu().numpy()
    unsolved_grid = predictions.reshape(9, 9).tolist()
    grid_to_solve = copy.deepcopy(unsolved_grid)
    solved_result = solve_sudoku(grid_to_solve)
    
    if not solved_result:
        raise HTTPException(status_code=400, detail="Board was read, but it is unsolvable. The image might be blurry.")
    
    warped_color = warp_perspective(image, board_contour)
    cell_h = warped_color.shape[0] // 9
    cell_w = warped_color.shape[1] // 9

    for r in range(9):
        for c in range(9):
            # draw number if empty (0)
            if unsolved_grid[r][c] == 0:
                text = str(solved_result[r][c])
                
                # center text inside cell
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = c * cell_w + (cell_w - text_size[0]) // 2
                text_y = r * cell_h + (cell_h + text_size[1]) // 2
                cv2.putText(warped_color, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', warped_color)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return {
        "original_grid": unsolved_grid,
        "solution": solved_result,
        "annotated_image": f"data:image/jpeg;base64,{encoded_img}"
    }
