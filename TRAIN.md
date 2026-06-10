# Training the Digit Recognition CNN

This project includes pre-trained weights for the Convolutional Neural Network so the app can run out-of-the-box. However, if you would like to generate a new dataset, tweak the model architecture, and train it from scratch, follow the guide below.

## Prerequisites

Ensure your Python virtual environment is activated and you have all the backend requirements installed:

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Generate Dataset

The CNN requires a diverse dataset of digits to accurately recognize numbers from various Sudoku boards.

Run the data generation script to build the training and validation sets:

```bash
cd backend/models
python3 generate_data.py
```

### Train the CNN

Once the dataset is generated, you can begin the training process. The PyTorch script will feed the data through the network, calculate the loss, and optimize the weights.

```bash
cd backend/models
python3 train.py
```

During training, the console will output the loss and accuracy metrics for each epoch. I plan to also only save the weights whenever the loss improves

### Verify Weights

When training is complete, the script will save the weights as a `.pth` file in `backend/models/saved_weights`. The FastAPI server is configured to load this file automatically upon startup.

You can now start your backend server, and it will use your trained model.
