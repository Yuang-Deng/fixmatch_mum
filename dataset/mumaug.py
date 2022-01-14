from re import S
import torch

class mumaug(object):
    def __init__(self, group_size=4, ng=4, nt=4):
        self.group_size = group_size
        self.ng = ng
        self.nt = nt
        self.mask = torch.zeros([group_size, ng, nt]).long()
        self.unmask = torch.zeros([group_size, ng, nt]).long()
        self.gen_num = 0
    
    def gen_mask(self):
        self.gen_num += 1
        for i in range(self.ng):
            for j in range(self.nt):
                indexes = torch.randperm(self.group_size)
                inverse_indexes = torch.zeros([self.group_size])
                for ind in range(self.group_size):
                    inverse_indexes[indexes[ind]] = ind
                self.mask[:, i, j] = indexes.long()
                self.unmask[:, i, j] = inverse_indexes.long()

    def imagemix(self, images):
        self.gen_mask()
        batch_size = images.size(0)
        imgw, imgh = images.size(2), images.size(3)
        blockw, blockh = imgw // self.ng, imgh // self.nt
        for i in range(self.ng):
            for j in range(self.nt):
                mask = torch.cat([self.mask[:, i, j] for _ in range(batch_size // self.group_size)])
                images[:, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh] = images[mask, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh]
        # for i in range(imgw):
        #     for j in range(imgw):
        #         mask = torch.cat([self.mask[:, i // blockw, j // blockh] for _ in range(batch_size // self.group_size)])
        #         images[:, :, i, j] = images[mask, :, i, j]
        return images

    def featureunmix(self, features):
        batch_size = features.size(0)
        imgw, imgh = features.size(2), features.size(3)
        blockw, blockh = imgw // self.ng, imgh // self.nt
        for i in range(self.ng):
            for j in range(self.nt):
                mask = torch.cat([self.unmask[:, i, j] for _ in range(batch_size // self.group_size)])
                features[:, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh] = features[mask, :, i * blockw: (i + 1) * blockw, j * blockh: (j + 1) * blockh]
        return features


# mask = torch.zeros([4, 4, 4])
# inverse_mask = torch.zeros([4, 4, 4])

# for i in range(4):
#     for j in range(4):
#         indexes = torch.randperm(4)
#         mask[:, i, j] = indexes
#         inverse_indexes = torch.zeros([4])
#         for ind in range(4):
#             inverse_indexes[indexes[ind]] = ind
#         inverse_mask[:, i, j] = inverse_indexes
# print(mask)
# print(inverse_mask)