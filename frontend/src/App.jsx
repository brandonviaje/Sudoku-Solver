import { useState } from 'react';
import axios from 'axios';
import SudokuGrid from './SudokuGrid';
import './App.css';

function App() {
  // UI state
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // backend state
  const [originalBoard, setOriginalBoard] = useState(null);
  const [solvedBoard, setSolvedBoard] = useState(null);
  const [annotatedImage, setAnnotatedImage] = useState(null);

  // handle file selection and reset previous results
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      
      // clear previous data
      setSolvedBoard(null);
      setOriginalBoard(null);
      setAnnotatedImage(null);
    }
  };

  // submit image to FastAPI backend
  const handleSolve = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await axios.post('http://localhost:8000/solve', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      // populate state when successful response data
      setOriginalBoard(response.data.original_grid);
      setSolvedBoard(response.data.solution);
      setAnnotatedImage(response.data.annotated_image);

    } catch (error) {
      console.error('Error solving puzzle:', error);
      alert('Failed to solve the puzzle. Make sure the backend is running!');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <div className="main-card">
        
        <header className="header">
          <h1 className="title">Sudoku Solver</h1>
          <p className="subtitle">Computer Vision & Backtracking</p>
        </header>
        
        <section className="upload-section">
          <label className={`file-upload-label ${previewUrl ? 'has-file' : ''}`}>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageChange} 
              className="hidden-input" 
            />
            <span className="upload-text">
              {selectedImage ? 'Swap Image' : 'Drop a puzzle here or click to browse'}
            </span>
          </label>

          {previewUrl && !solvedBoard && (
            <div className="preview-container fade-in">
              <img src={previewUrl} alt="Sudoku Preview" className="image-preview" />
            </div>
          )}
        </section>

        <button 
          className={`solve-button ${isLoading ? 'loading' : ''}`} 
          onClick={handleSolve} 
          disabled={!selectedImage || isLoading}
        >
          {isLoading ? <span className="spinner"></span> : 'Solve Puzzle'}
        </button>

        {/* result section */}
        {solvedBoard && originalBoard && (
          <main className="results-container fade-in">
            
            <section className="photo-comparison-row">
              <div className="result-card">
                <h2>Original Photo</h2>
                <div className="fixed-image-wrapper">
                  <img src={previewUrl} alt="Original Upload" className="fixed-image" />
                </div>
              </div>

              {annotatedImage && (
                <div className="result-card">
                  <h2>Completed Photo</h2>
                  <div className="fixed-image-wrapper">
                    <img src={annotatedImage} alt="Solved Sudoku" className="fixed-image" />
                  </div>
                </div>
              )}
            </section>

            <section className="digital-boards-row">
              <div className="result-card">
                <h2>Model Detected</h2>
                <SudokuGrid board={originalBoard} />
              </div>
              
              <div className="result-card">
                <h2>Solution</h2>
                <SudokuGrid board={solvedBoard} />
              </div>
            </section>

          </main>
        )}
        
      </div>
    </div>
  );
}

export default App;
