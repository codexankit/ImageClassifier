import numpy as np
from PIL import Image
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from data_utility import process_image

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

def build_network(architecture, hidden_units):
    
    print("==============================================================================================")
    print(f"Building the Model || Architecture: {architecture} || Hidden units: {hidden_units}")
    
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_units = 1024
        
    elif architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_units = 2208
        
    elif architecture == 'densenet169':
        model = models.densenet169(pretrained=True)
        input_units = 1664
        
    elif architecture == 'densenet201':
        model = models.densenet201(pretrained=True)
        input_units = 1920
        
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units = 25088
        
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_units = 25088
        
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_units = 9216
        
    
    # Defining the classifier (the fully connected layer)
    
    classifier = nn.Sequential(
              nn.Linear(input_units, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
    
    model.classifier = classifier
    
    print("Model Built successfully!")
    print("=============================================================================================================")
    
    return model

def train_network(model, epochs, learning_rate, trainloader, validloader, gpu):
    
    print("=============================================================================================================")
    print(f"Training in process || Model: {model} || Epochs: {epochs} || Learning Rate: {learning_rate} || GPU: {gpu}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Store the losses for plotting the graph
    training_loss, validation_loss, validation_accuracy = [], [], []
    
    steps = 0
    running_loss = 0
    print_every = 10
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                training_loss.append(running_loss/print_every)
                validation_loss.append(valid_loss/len(validloader))
                validation_accuracy.append(accuracy/len(validloader))
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
               
    print("Model training finished!")
    print("======================================================================================================")
    
#     plt.plot(training_loss, label='Training loss')
#     plt.plot(validation_loss, label='Validation loss')
#     plt.plot(validation_accuracy, label='Validation accuracy')
#     plt.legend(frameon=False)

    return model, criterion


def evaluate_model(model, testloader, criterion, gpu):
    
    print("======================================================================================================")
    print("Model validation on the test set")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loss = 0
    test_accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            print(f"Test loss: {test_loss/len(testloader):.3f}, "
                f"Test accuracy: {test_accuracy/len(testloader):.3f}")
            
            running_loss = 0
            
            print("Model validation done.")
            print("============================================================================================")
            
            
def save_model(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    
    print("=============================================================================================")
    print("Saving the model.")
    
    checkpoint = {'model_state_dict': model.state_dict(),
                  'architecture': architecture,
                  'hidden_units': hidden_units,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'class_to_idx': model.class_to_idx
    }
    
    checkpoint_path = save_dir + "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    
    print("Model saved to {}".format(checkpoint_path))
    print("=============================================================================================")
    
def load_checkpoint(filepath):
    
    print("Loading checkpoint...")
    checkpoint = torch.load(filepath)
    model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print("Checkpoint loaded successfully.")
    print("=============================================================================================")
    
    return model

def predict(processed_image, model, topk): 
    model.eval()
    with torch.no_grad():
        logps = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probs.numpy()[0], classes
                  



        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        