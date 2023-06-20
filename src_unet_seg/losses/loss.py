
import torch
from losses.lovasz_loss import lovasz_hinge


def dice_loss(logits, target):
    smooth = 1.
    prob  = torch.sigmoid(logits)
    batch = prob.size(0)
    prob   = prob.view(batch,1,-1)
    target = target.view(batch,1,-1)
    
    intersection = torch.sum(prob*target, dim=2)
    denominator  = torch.sum(prob, dim=2) + torch.sum(target, dim=2)

    dice = (2*intersection + smooth) / (denominator + smooth)
    dice = torch.mean(dice)
    dice_loss = 1. - dice
    return dice_loss





import torch
import torch.nn.functional as F

# def dice_loss_non_empty_masks(prediction, target):
#     """
#     Calculates the Dice Loss for non-empty masks.
    
#     Args:
#         prediction (torch.Tensor): Predicted mask tensor with shape (batch_size, num_channels, height, width)
#         target (torch.Tensor): Target mask tensor with shape (batch_size, num_channels, height, width)
        
#     Returns:
#         torch.Tensor: Dice Loss for non-empty masks
#     """
#     batch_size, num_channels, height, width = prediction.size()

#     # Flatten the tensors
#     prediction = prediction.view(batch_size, num_channels, -1)
#     target = target.view(batch_size, num_channels, -1)
    
#     # Calculate the intersection and union of the non-empty masks
#     intersection = torch.sum(prediction * target, dim=2)  # shape: (batch_size, num_channels)
#     target_sum = torch.sum(target, dim=2)  # shape: (batch_size, num_channels)
#     prediction_sum = torch.sum(prediction, dim=2)  # shape: (batch_size, num_channels)
    
#     # Exclude non-empty masks
#     mask_indices = torch.nonzero(target_sum > 0, as_tuple=True)
#     intersection = intersection[mask_indices]
#     target_sum = target_sum[mask_indices]
#     prediction_sum = prediction_sum[mask_indices]
    
#     # Calculate the Dice Loss
#     dice = (2 * intersection) / (target_sum + prediction_sum + 1e-8)  # add epsilon for numerical stability
#     dice_loss = 1 - torch.mean(dice)
    
#     return dice_loss


def dice_loss(prediction, target):
    """
    Calculates the Dice Loss.
    
    Args:
        prediction (torch.Tensor): Predicted mask tensor with shape (batch_size, num_channels, height, width)
        target (torch.Tensor): Target mask tensor with shape (batch_size, num_channels, height, width)
        
    Returns:
        torch.Tensor: Dice Loss
    """
    batch_size, num_channels, height, width = prediction.size()

    # Flatten the tensors
    prediction = prediction.view(batch_size, num_channels, -1)
    target = target.view(batch_size, num_channels, -1)
    
    # Calculate the intersection and union of the masks
    intersection = torch.sum(prediction * target, dim=2)  # shape: (batch_size, num_channels)
    target_sum = torch.sum(target, dim=2)  # shape: (batch_size, num_channels)
    prediction_sum = torch.sum(prediction, dim=2)  # shape: (batch_size, num_channels)
    
    # Calculate the Dice Loss
    dice = (2 * intersection) / (target_sum + prediction_sum + 1e-8)  # add epsilon for numerical stability
    dice_loss = 1 - torch.mean(dice)
    
    return dice_loss


def dice_loss_non_empty_masks(prediction, target):
    """
    Calculates the Dice Loss for non-empty masks.
    
    Args:
        prediction (torch.Tensor): Predicted mask tensor with shape (batch_size, num_channels, height, width)
        target (torch.Tensor): Target mask tensor with shape (batch_size, num_channels, height, width)
        
    Returns:
        torch.Tensor: Dice Loss for non-empty masks
    """
    batch_size, num_channels, height, width = prediction.size()

    # Flatten the tensors
    prediction = prediction.view(batch_size, num_channels, -1)
    target = target.view(batch_size, num_channels, -1)
    
    # Calculate the intersection and union of the non-empty masks
    intersection = torch.sum(prediction * target, dim=2)  # shape: (batch_size, num_channels)
    target_sum = torch.sum(target, dim=2)  # shape: (batch_size, num_channels)
    prediction_sum = torch.sum(prediction, dim=2)  # shape: (batch_size, num_channels)
    
    # Exclude non-empty masks
    mask_indices_non_empty = torch.nonzero(target_sum > 0, as_tuple=True)
    mask_indices_empty = torch.nonzero(target_sum <= 0, as_tuple=True)

    intersection_non_empty = intersection[mask_indices_non_empty]
    target_sum_non_empty = target_sum[mask_indices_non_empty]
    prediction_sum_non_empty = prediction_sum[mask_indices_non_empty]
    
    # Calculate the Dice Loss
    dice_non_empty = (2 * intersection_non_empty) / (target_sum_non_empty + prediction_sum_non_empty + 1e-8)  # add epsilon for numerical stability
    dice_loss_non_empty = 1 - torch.mean(dice_non_empty)
    

    intersection_empty = intersection[mask_indices_empty]
    target_sum_empty = target_sum[mask_indices_empty]
    prediction_sum_empty = prediction_sum[mask_indices_empty]
    
    # Calculate the Dice Loss
    dice_empty = (2 * intersection_empty) / (target_sum_empty + prediction_sum_empty + 1e-8)  # add epsilon for numerical stability
    dice_loss_empty = 1 - torch.mean(dice_empty)
    
    return dice_loss_non_empty, dice_loss_empty





def criterion_lovasz_hinge_non_empty(criterion, logits_deep, y):
    batch,c,h,w = y.size()
    y2 = y.view(batch*c,-1)
    logits_deep2 = logits_deep.view(batch*c,-1)
    
    y_sum = torch.sum(y2, dim=1)
    non_empty_idx = (y_sum!=0)
    
    if non_empty_idx.sum()==0:
        return torch.tensor(0)
    else:
        loss  = criterion(logits_deep2[non_empty_idx], 
                          y2[non_empty_idx])
        loss += lovasz_hinge(logits_deep2[non_empty_idx].view(-1,h,w), 
                             y2[non_empty_idx].view(-1,h,w))
        return loss