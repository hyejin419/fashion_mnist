import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 데이터 불러오기
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. 모델 정의 (간단한 CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             
            nn.Flatten(),                   
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Linear(128, 10)               
        )

    def forward(self, x):
        return self.model(x)
    

# 3. 장비 설정 (GPU 사용 가능하면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 4. 학습 루프
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


# 5. 테스트 정확도 측정
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f" 테스트 정확도: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'fashion_weights.pth')  # 가중치 저장
torch.save(model, 'fashion.pt')  # 모델 자체를 저장