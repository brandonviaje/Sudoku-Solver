import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from cnn_build import SudokuCNN

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # data prep
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "dataset")
    
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # 80% training / 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model, loss, and optimizer
    model = SudokuCNN().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #  training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            # track accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train

        # validation
        model.eval() 
        correct_val = 0
        total_val = 0

        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Acc: {val_accuracy:.2f}%")

    # save trained weights
    os.makedirs(os.path.join(BASE_DIR, "saved_weights"), exist_ok=True)
    save_path = os.path.join(BASE_DIR, "saved_weights", "sudoku_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete! Weights saved to: {save_path}")

if __name__ == "__main__":
    train_model()
