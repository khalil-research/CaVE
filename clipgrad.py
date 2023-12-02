import torch

def clipGrad(model, threshold):
    for param in model.parameters():
        if param.grad is not None:
            with torch.no_grad():
                grad = param.grad
                grad[torch.abs(grad) < threshold] = 0
