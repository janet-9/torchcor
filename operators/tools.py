class Normalizer:
    def __init__(self, scale_range=(0, 1)):

        self.scale_range = scale_range
        self.normalization_params = None

        self.y_min = None
        self.y_max = None

    def normalize(self, y):
        # self.y_min = y.min()
        # self.y_max = y.max()

        # Normalize y to the desired range [scale_range[0], scale_range[1]]
        if self.scale_range == (0, 1):
            normalized_y = (y - self.y_min) / (self.y_max - self.y_min)
        elif self.scale_range == (-1, 1):
            normalized_y = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        
        return normalized_y

    def inverse_normalize(self, normalized_y):
        if self.scale_range == (0, 1):
            original_y = normalized_y * (self.y_max - self.y_min) + self.y_min
        elif self.scale_range == (-1, 1):
            original_y = (normalized_y + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        
        return original_y
