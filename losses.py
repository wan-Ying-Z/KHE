import torch
from torch import nn


class CrossEntropy(nn.Module):

    def forward(self, input, target):#计算交叉熵损失的过程
        scores = torch.sigmoid(input)
        #print('score',scores)

        target_active = (target == 1).float()  # from -1/1 to 0/1
        loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
        missing_values_mask = (target != 0).float()
        #最后返回的时候求了平均交叉熵
        #print('loss_item:',loss_terms)
        b=loss_terms.sum()/len(loss_terms)
        #a=(loss_terms * missing_values_mask).sum() / missing_values_mask.sum()
        #print('a:',a)
        #print('avg_Loss_item:',b)
        return b

class muti_CrossEntropy(nn.Module):

    def forward(self, output,target):
        output = output.detach().numpy()
        target = target.detach().numpy()

        scores = []
        losses = torch.zeros([663, 1])
        length=target.size(1)
        for i in range(length):
            x = target[:, i]
            x = torch.from_numpy(x)

            y = output[:, i]
            y = torch.from_numpy(y)

            score = torch.sigmoid(y)
            scores.append(score)
            target_active = (x == 1).float()
            loss_terms = -(target_active * torch.log(score) + (1 - target_active) * torch.log(1 - score))
            b = loss_terms.sum() / len(loss_terms)
            losses[i] = b

        loss = losses.mean(0).requires_grad_()
        return loss



LOSS_FUNCTIONS = {
    'CrossEntropy': CrossEntropy(),#交叉熵损失函数
    'muti_CrossEntropy':muti_CrossEntropy(),
    'MSE': nn.MSELoss()#均方误差
}


