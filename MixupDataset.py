from torch.utils.data import Dataset
import random


class MixupDataset(Dataset):
    def __init__(self, base_ds, mixup_prob):
        super(MixupDataset, self).__init__()
        self.base = base_ds
        self.mixup_prob = mixup_prob
        self.label_map = {}
        self.__build_label_map()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        (x, y) = self.base[item]
        p = random.random()
        if p >= self.mixup_prob:
            mix_x, mix_y = self.sample(y)
            x = 0.5 * (x + mix_x)
            y = y if random.random() < 0.5 else mix_y
        return x, y

    def sample(self, y):
        mix_y = random.sample(list(self.label_map.keys() - {y}), 1)[0]
        mix_x = random.sample(self.label_map[mix_y], 1)[0]
        return mix_x, mix_y

    def __build_label_map(self):
        for (x, y) in self.base:
            if y not in self.label_map:
                self.label_map[y] = []
            self.label_map[y].append(x)