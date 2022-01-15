from re import S
import torch

def gen_mask(ng, nt, group_size):
    mask = torch.zeros([group_size, ng, nt]).long()
    unmask = torch.zeros([group_size, ng, nt]).long()
    for i in range(ng):
        for j in range(nt):
            indexes = torch.randperm(group_size)
            inverse_indexes = torch.zeros([group_size])
            for ind in range(group_size):
                inverse_indexes[indexes[ind]] = ind
            mask[:, i, j] = indexes.long()
            unmask[:, i, j] = inverse_indexes.long()
    return mask, unmask

class mumaug(object):
    def __init__(self, group_size=4, ng=4, nt=4):
        self.group_size = group_size
        self.ng = ng
        self.nt = nt
        self.mask = torch.zeros([0, ng, nt]).long()
        self.unmask = torch.zeros([0, ng, nt]).long()
        self.gen_num = 0
    
    # def gen_mask(self):
    #     self.gen_num += 1
    #     for i in range(self.ng):
    #         for j in range(self.nt):
    #             indexes = torch.randperm(self.group_size)
    #             inverse_indexes = torch.zeros([self.group_size])
    #             for ind in range(self.group_size):
    #                 inverse_indexes[indexes[ind]] = ind
    #             self.mask[:, i, j] = indexes.long()
    #             self.unmask[:, i, j] = inverse_indexes.long()

    def imagemix(self, images):
        self.mask = torch.zeros([0, self.ng, self.nt]).long()
        self.unmask = torch.zeros([0, self.ng, self.nt]).long()
        base = torch.ones([self.group_size, self.ng, self.nt]).long()
        batch_size = images.size(0)
        imgw, imgh = images.size(2), images.size(3)
        blockw, blockh = imgw // self.ng, imgh // self.nt
        for gindex in range(batch_size // self.group_size):
            mask, unmask = gen_mask(self.ng, self.nt, self.group_size)
            self.mask = torch.cat([self.mask, (mask + (base * gindex * 4))])
            self.unmask = torch.cat([self.unmask, (unmask + (base * gindex * 4))])
        for i in range(self.ng):
            for j in range(self.nt):
                mask = self.mask[:, i, j]
                images[:, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh] = images[mask, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh]
        return images

    def featureunmix(self, features):
        imgw, imgh = features.size(2), features.size(3)
        blockw, blockh = imgw // self.ng, imgh // self.nt
        for i in range(self.ng):
            for j in range(self.nt):
                mask = self.unmask[:, i, j]
                features[:, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh] = features[mask, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh]
        return features