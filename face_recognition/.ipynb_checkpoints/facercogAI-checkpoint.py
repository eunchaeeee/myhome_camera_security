import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

class SpecificFolder(datasets.ImageFolder):
    def select_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        classes = [cls for cls in classes if cls != ".ipynb_checkpoints"]
        class_to_idx = {cls: idx for cls, idx in class_to_idx.items() if cls != ".ipynb_checkpoints"}
        return classes, class_to_idx

class IMGclassifier(object):
    def __init__(self, dir_path):
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                            ])
        self.dataset = SpecificFolder(dir_path, transform = self.transform)
        self.model_setting()  
        self.preprocess(dir_path)
        self.trained_model_name = 'namedtrained_model.pth'

    def preprocess(self, dir_path):
        train_idx, val_idx = train_test_split(list(range(len(self.dataset))), test_size=0.2, random_state=42)
        train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    def model_setting(self, num_classes=None):
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        if num_classes is None:
            num_classes = len(self.dataset.classes)
        self.model.fc = nn.Linear(num_features, num_classes) 
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def model_training(self):
        num_epochs = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
        
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
            self.model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
            val_loss = val_loss / len(self.val_loader.dataset)
            val_acc = correct / len(self.val_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        
        self.save_model(self.model)
    
    def save_model(self, model):
        torch.save(model.state_dict(), self.trained_model_name)
        
    def load_model(self):
        self.model_setting(num_classes=1)
        state_dict = torch.load(self.trained_model_name)
        
        fc_weight_key = 'fc.weight'
        fc_bias_key = 'fc.bias'
        num_classes_in_checkpoint = state_dict[fc_weight_key].size(0)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes_in_checkpoint)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_image(self, image, model, transform, class_names, threshold=0.9):
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device)
        model.to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)  # 모델의 출력에 소프트맥스 적용
            max_prob, predicted = torch.max(probabilities, 1)
    
            # 일정 임계값(threshold)보다 낮으면 'unknown' 반환
            if max_prob.item() < threshold:
                return 'unknown'
            else:
                return class_names[predicted.item()]
    
    def run_prediction(self, image):
        if not os.path.isfile(self.trained_model_name):
            self.model_training()
        else:
            self.load_model()
        
        class_names = self.dataset.classes
        predicted_class = self.predict_image(image, self.model, self.transform, class_names)
        print(f'Predicted class: {predicted_class}')
        return predicted_class



