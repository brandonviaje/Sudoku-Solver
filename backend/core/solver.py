from typing import List

"""
Return box index for sudoku board
"""
def box_index(r, c):
    return (r // 3) * 3 + (c // 3)

"""
Get all valid numbers you can place in current cell
"""
def get_candidates(r, c, rows, cols, boxes):
    return {
        num for num in range(1, 10)
        if num not in rows[r]
        and num not in cols[c]
        and num not in boxes[box_index(r, c)]
    }

"""
Backtracking Algorithm for Sudoku Solver
"""
def backtrack(board, empty_cells, rows, cols, boxes):
    # base case: no more empty cells
    if not empty_cells:
        return True

    # find cell with fewest options
    min_options = 10
    best_idx = -1
    best_candidates = None

    for i, (r, c) in enumerate(empty_cells):
        candidates = get_candidates(r, c, rows, cols, boxes)
        if len(candidates) < min_options:
            min_options = len(candidates)
            best_idx = i
            best_candidates = candidates

        if min_options == 1:
            break
    
    if min_options == 0:
        return False

    # pick best cell
    r, c = empty_cells.pop(best_idx)

    for num in best_candidates:
        # place
        board[r][c] = num
        rows[r].add(num)
        cols[c].add(num)
        boxes[box_index(r, c)].add(num)

        # recurse further
        if backtrack(board, empty_cells, rows, cols, boxes):
            return True

        # undo/backtrack
        board[r][c] = 0
        rows[r].remove(num)
        cols[c].remove(num)
        boxes[box_index(r, c)].remove(num)

    # restore if failed
    empty_cells.insert(best_idx, (r, c))
    return False

"""
Main Entry Point for Sudoku Solver
"""
def solve_sudoku(board: List[List[int]]):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty_cells = []

    # initialize sets
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == 0:
                empty_cells.append((r, c))
            else:
                if val in rows[r] or val in cols[c] or val in boxes[box_index(r, c)]:
                    return False
                rows[r].add(val)
                cols[c].add(val)
                boxes[box_index(r, c)].add(val)

    backtrack(board, empty_cells, rows, cols, boxes)
    return board

if __name__ == "__main__":
    
    grid = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 6, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ]

    print(f"Before: {grid}")
    grid = solve_sudoku(grid)
    print(f"After: {grid}")

