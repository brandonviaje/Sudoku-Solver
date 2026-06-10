<div align="center">

# Sudoku Solver

<img width="530" height="700" alt="Image" src="https://github.com/user-attachments/assets/a0482015-1f6d-4344-9c0f-0695dd90e31d" />

> A web application that takes a raw image of an unsolved Sudoku board, visually processes the grid using a CNN, and calculates the solution. 
</div>

## Features

* **Image Processing:** Uses OpenCV for perspective warping, grayscale conversion, and Gaussian filtering to isolate the grid.
* **Digit Recognition:** Custom Convolutional Neural Network (CNN) built with PyTorch to read handwritten and printed digits.
* **Algorithmic Solving:** Implements an optimized backtracking algorithm to compute the final puzzle solution instantly.
* **Modern Interface:** Features a sleek, responsive dark-mode UI built with React.

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend** | React, Node.js, HTML/CSS |
| **Backend** | Python, FastAPI |
| **AI & Vision** | PyTorch, OpenCV |
| **Deployment** | Vercel (Frontend), Render (Backend) |

---
# How to Run 

## Locally

Follow these steps to run both the API and the user interface on your local machine.

### Clone the repo

``` bash
git clone https://github.com/brandonviaje/Sudoku-Solver.git
```

### Start the Backend

Open a terminal and navigate to the backend directory to set up your Python environment

``` bash
cd ~/dev/Sudoku-Solver/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Start the FrontEnd

In a seperate terminal, run the frontend. Ensure you have **Node.js** on your computer

``` bash
cd ~/dev/Sudoku-Solver/frontend
npm install
npm run dev
```

## Deployed

This is also deployed on a webpage found here (link).
