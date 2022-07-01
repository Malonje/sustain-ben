import torch.nn.functional as F
from sklearn.metrics import r2_score
import numpy as np

def l1_l2_loss(pred, true, l1_weight, scores_dict):
    """
    Regularized MSE loss; l2 loss with l1 loss too.

    Parameters
    ----------
    pred: torch.floatTensor
        The model predictions
    true: torch.floatTensor
        The true values
    l1_weight: int
        The value by which to weight the l1 loss
    scores_dict: defaultdict(list)
        A dict to which scores can be appended.

    Returns
    ----------
    loss: the regularized mse loss
    """
    loss = F.mse_loss(pred, true)

    scores_dict["l2"].append(loss.item())

    if l1_weight > 0:
        l1 = F.l1_loss(pred, true)
        loss += l1
        scores_dict["l1"].append(l1.item())
    scores_dict["loss"].append(loss.item())


    true = true.detach().cpu().numpy().flatten()
    pred = pred.detach().cpu().numpy().flatten()
    assert (true.shape == pred.shape)

    error = pred-true
    RMSE = np.sqrt(np.mean(error**2))
    R2 = r2_score(true, pred)
    scores_dict["RMSE"].append(RMSE)
    return loss, scores_dict
