 # 1 Tensors (specialized data structures for efficient computation) , In pytorch we have tensors which are similar to numpy arrays but with additional capabilities for GPU acceleration and automatic differentiation.
import torch
import numpy as np
import pandas as pd
# 2 Autograd (automatic differentiation for gradient computation) , PyTorch provides an autograd system that automatically computes gradients for tensor operations, which is essential for training neural networks.
data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)

x_data = torch.tensor(data)

# from numpy arrays
numpy_array = np.array([[1, 2], [3, 4]])
torch_tensor = torch.from_numpy(numpy_array)

# from another tensor
x1 = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x1} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# Joining tensors You can use torch.cat to concatenate a sequence of tensors along a given dimension
df = torch.cat((x_data, x_rand, x1), dim=1)
print(f"Concatenated Tensor: \n {df} \n")

# torch.autograd is PyTorch’s automatic differentiation engine that powers neural network training. In this section, you will get a conceptual understanding of how autograd helps a neural network train.
# Neural networks are collections of nested functions. For example, a simple feedforward neural network can be thought of as a composition of linear transformations and non-linear activation functions. When you perform operations on tensors that have requires_grad=True, PyTorch builds a computational graph in the background. This graph tracks all the operations performed on the tensors, allowing PyTorch to compute gradients automatically when you call backward() on the final output tensor.

# Neural Networks works in two phases: the forward pass and the backward pass. During the forward pass, you compute the output of the network given some input data. During the backward pass, you compute the gradients of the loss with respect to each parameter in the network, which is essential for updating the parameters during training.

# lets take a look of a pretrained resnet18 model and see how autograd works in it
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
# Load a pretrained ResNet18 model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# Set the model to evaluation mode  
data = torch.randn(1, 3, 224, 224)  # Example input tensor (batch size of 1, 3 color channels, 224x224 image)
# Target labels (Generate random class 0-999)
target = torch.randint(0, 1000, (1,), dtype=torch.long) 

# Use CrossEntropyLoss for classification
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(f"Target Class Index: {target.item()}")

# Training loop to demonstrate learning (overfitting to single batch)
print("\nStarting Training Loop...")
gradient_data = []

for epoch in range(100):
    # Forward pass
    optimizer.zero_grad()
    output = model(data)
    
    # Compute Loss
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Store gradients for the first epoch to show structure
    if epoch == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_data.append({'Layer': name, 'Gradient': param.grad.norm().item()})
        df = pd.DataFrame(gradient_data)
        print("\nGradient DataFrame (Epoch 0):")
        print(df.head()) # proper display of head
        print("...\n")

    # Update weights
    optimizer.step()
    
    # Check accuracy
    prediction = output.argmax(dim=1)
    accuracy = (prediction == target).float().mean()
    
    if (epoch + 1) % 10 == 0:
         print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Prediction: {prediction.item()}, Accuracy: {accuracy.item() * 100:.2f}%")

print(f"\nFinal Prediction: {output.argmax(dim=1).item()}")
print(f"Target Label: {target.item()}")    

