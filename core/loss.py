import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()

        #penalty for getting a negative category wrong
        self.gamma_neg = gamma_neg
        #penalty for getting a positive category wrong
        self.gamma_pos = gamma_pos
        #confidence threshold
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):

        #(positive labels) probabilities that the product belong to each category
        xs_p = torch.sigmoid(logits)

        #(negative labels) the inverse probability of the above (q=1-p)
        xs_n = 1.0 - xs_p

 
        if self.clip and self.clip > 0:
            xs_n = (xs_n + self.clip).clamp(max=1.0)


        loss_pos = targets * torch.log(xs_p.clamp(min=self.eps)) * (1.0 - xs_p).pow(self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(xs_n.clamp(min=self.eps)) * (1.0 - xs_n).pow(self.gamma_neg)
        
        return -(loss_pos + loss_neg).mean()

