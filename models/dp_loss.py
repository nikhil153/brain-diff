import torch.nn as nn

def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    #print(loss)
    return loss

def twin_KLDivLoss(x1,x2, y1,y2):
    """Returns K-L Divergence loss for twin arch
    """
    loss1 = my_KLDivLoss(x1,y1)
    loss2 = my_KLDivLoss(x2,y2)
    loss = 0.5*(loss1 + loss2)
    return loss