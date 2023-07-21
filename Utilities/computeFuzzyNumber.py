import torch

def computeFuzzyNumber(trainset, delta):
    if trainset.size(0) == 0:
        print('The input dataset is null!')
        
    else:
        row1, col1 = trainset.size()
        col1 -= 1  # Remove the last column (labels)

        group1 = trainset[trainset[:, col1] == -1, :]
        group2 = trainset[trainset[:, col1] == 1, :]

        row_g1 = group1.size(0)
        row_g2 = group2.size(0)

        mean_g1 = torch.mean(group1[:, :col1], dim=0)
        mean_g2 = torch.mean(group2[:, :col1], dim=0)

        max_g1 = 0
        max_g2 = 0

        for i in range(row_g1):
            distance_g1 = torch.norm(group1[i, :col1] - mean_g1)
            if distance_g1 >= max_g1:
                max_g1 = distance_g1

        for j in range(row_g2):
            distance_g2 = torch.norm(group2[j, :col1] - mean_g2)
            if distance_g2 >= max_g2:
                max_g2 = distance_g2

        fms = torch.zeros(row1, 1)
        for i in range(row1):
            if trainset[i, col1] == -1:
                fms[i, 0] = 1 - (torch.sqrt(torch.norm(trainset[i, :col1] - mean_g1)) / (max_g1 + delta))
            if trainset[i, col1] == 1:
                fms[i, 0] = 1 - (torch.sqrt(torch.norm(trainset[i, :col1] - mean_g2)) / (max_g2 + delta))

        return fms


def computeFuzzyNew(trainSetKernel, w, b):
    eta = 1e-3  # eta=0.001
    if trainSetKernel.size(0) == 0:
        print('The input dataset is null!')
        return None
    else:
        row1, col1 = trainSetKernel.size()
        col1 -= 1  # Remove the last column (labels)
        labels = torch.sort(torch.unique(trainSetKernel[:, col1])).values

        fms = torch.zeros(row1, 1)
        group1 = trainSetKernel[trainSetKernel[:, col1] == labels[0], :]
        group1Ex = trainSetKernel[trainSetKernel[:, col1] != labels[0], :]

        class1Num = group1.size(0)
        class1NumEx = group1Ex.size(0)

        # For binary classification problem, group1Ex is equivalent to the second class.
        dist1 = torch.abs(torch.mm(group1, w.view(-1, 1)) + b) / torch.sqrt(torch.norm(w))
        dist2 = torch.abs(torch.mm(group1Ex, w.view(-1, 1)) + b) / torch.sqrt(torch.norm(w))

        maxDist1 = torch.max(dist1)
        maxDist2 = torch.max(dist2)

        for i in range(row1):
            if trainSetKernel[i, col1] == labels[0]:
                fms[i, 0] = dist1[i] / (maxDist1 + eta)
            else:
                fms[i, 0] = dist2[i] / (maxDist2 + eta)

        return fms
