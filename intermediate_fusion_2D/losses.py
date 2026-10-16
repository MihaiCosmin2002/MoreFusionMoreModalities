import torch
import torch.nn.functional as F

ALPHA = 0.25
GAMMA = 1

def dice_coef(y_true, y_pred, smooth=1e-9):
    """
    PyTorch version of dice_coef.
    Note: y_true and y_pred should be flattened for the calculation.
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    
    intersection = torch.sum(y_true_f * y_pred_f)
    
    # Matching your exact formula: (2*intersection) / (sum(true^2) + sum(pred^2))
    return (2. * intersection + smooth) / (torch.sum(y_true_f**2) + torch.sum(y_pred_f**2) + smooth)

def dice_loss(y_true, y_pred):
    """
    PyTorch version of dice_loss.
    """
    return torch.mean(1 - dice_coef(y_true, y_pred))

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    """
    PyTorch version of FocalLoss.
    """
    inputs = torch.flatten(inputs)
    targets = torch.flatten(targets)
    
    # PyTorch's binary_cross_entropy requires inputs to be in range [0, 1]
    BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
    BCE_EXP = torch.exp(-BCE)
    
    focal_loss = torch.mean(alpha * torch.pow((1 - BCE_EXP), gamma) * BCE)
    
    return focal_loss

def DiceBCELoss(y_true, y_pred, smooth=1e-9):
    """
    PyTorch version of DiceBCELoss combining Dice and Binary Cross Entropy.
    """
    
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred) # These are raw 'logits'
    
    # For Dice, we manually apply sigmoid to get probabilities [0, 1]
    y_pred_prob = torch.sigmoid(y_pred_f)
    
    intersection = torch.sum(y_true_f * y_pred_prob)
    dice_loss_val = 1 - ((2. * intersection + smooth) / 
                         (torch.sum(y_true_f**2) + torch.sum(y_pred_prob**2) + smooth))

    # For BCE, we use 'with_logits' which is safe for autocast
    # Note: Input (y_pred_f) comes first in PyTorch
    BCE = F.binary_cross_entropy_with_logits(y_pred_f, y_true_f, reduction='mean')
    
    return BCE + dice_loss_val

def L2Loss(y_true, y_pred):
    """
    PyTorch version of L2Loss (Mean Squared Error).
    """
    inputs = torch.flatten(y_pred)
    targets = torch.flatten(y_true)

    # Element-wise squared difference
    L2_Loss = (inputs - targets)**2

    return L2_Loss

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)