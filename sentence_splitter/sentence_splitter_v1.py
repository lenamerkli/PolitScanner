from torch.nn import Module, Linear, GELU, ModuleList
from torch import Tensor, zeros, abs


INPUT_SIZE = 1024
HIDDEN_SIZES = [1536, 1024, 512]
# HIDDEN_SIZES = [4096, 2048, 1024, 512]
OUTPUT_SIZE = 64
PARAMETERS = INPUT_SIZE * HIDDEN_SIZES[0] + sum((HIDDEN_SIZES[i - 1] * HIDDEN_SIZES[i] for i in range(1, len(HIDDEN_SIZES) - 1))) + HIDDEN_SIZES[-1] * OUTPUT_SIZE


class SentenceSplitter(Module):
    def __init__(self):
        super(SentenceSplitter, self).__init__()
        self.layers = ModuleList()
        self.gelu = GELU()
        self.layers.append(Linear(INPUT_SIZE, HIDDEN_SIZES[0]))
        for i in range(1, len(HIDDEN_SIZES)):
            self.layers.append(Linear(HIDDEN_SIZES[i - 1], HIDDEN_SIZES[i]))
        self.layers.append(Linear(HIDDEN_SIZES[-1], OUTPUT_SIZE))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        for i in range(len(self.layers)):
            try:
                x = self.layers[i](x)
                if i != len(self.layers) - 1:
                    x = self.gelu(x)
            except Exception as e:
                e.add_note(f"Error in layer {i}")
                raise e
        x = x.view(x.size(0), -1)
        return x


class SentenceSplitterLoss(Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super(SentenceSplitterLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        losses = zeros(output.shape[0], device=output.device)
        batched = len(output.shape) > 1
        if not batched:
            output = output.unsqueeze(0)
            target = target.unsqueeze(0)
        delta = abs(output - target)
        for i in range(delta.shape[1]):
            if delta[0, i] >= 0.5:
                losses += self._alpha * delta[0, i] + self._beta
        return losses


if __name__ == "__main__":
    model = SentenceSplitter()
    print(model.layers)
