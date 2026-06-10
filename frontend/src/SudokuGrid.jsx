import React from 'react';
import './SudokuGrid.css';

const SudokuGrid = ({ board }) => {
  if (!board || board.length === 0) return null;

  return (
    <div className="sudoku-grid">
      {board.map((row, rIndex) => (
        <div key={`row-${rIndex}`} className="sudoku-row">
          {row.map((cell, cIndex) => {
            // calculate grid boundaries for 3x3 border
            let cellClasses = "sudoku-cell";
            if (cIndex % 3 === 2 && cIndex !== 8) cellClasses += " border-right";
            if (rIndex % 3 === 2 && rIndex !== 8) cellClasses += " border-bottom";
            
            return (
              <div key={`cell-${rIndex}-${cIndex}`} className={cellClasses}>
                {cell !== 0 ? cell : ''}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

export default SudokuGrid;
