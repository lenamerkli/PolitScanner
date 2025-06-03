from torch.nn import Module, Linear, GELU
from torch import Tensor, zeros, abs
from bitsandbytes.nn import Linear8bitLt
import typing as t


INPUT_SIZE = 2048
HIDDEN_SIZES = [2048, 1536, 1024]
OUTPUT_SIZE = 2 * 256
PARAMETERS = INPUT_SIZE * HIDDEN_SIZES[0] + sum((HIDDEN_SIZES[i - 1] * HIDDEN_SIZES[i] for i in range(1, len(HIDDEN_SIZES) - 1))) + HIDDEN_SIZES[-1] * OUTPUT_SIZE


class SentenceSplitter(Module):
    def __init__(self):
        super(SentenceSplitter, self).__init__()
        self.layers: t.List[t.Union[Linear, Linear8bitLt]] = []
        self.gelu = GELU()
        self.layers.append(Linear(INPUT_SIZE, HIDDEN_SIZES[0]))
        for i in range(1, len(HIDDEN_SIZES) - 1):
            self.layers.append(Linear8bitLt(HIDDEN_SIZES[i - 1], HIDDEN_SIZES[i]))
        self.layers.append(Linear(HIDDEN_SIZES[-1], OUTPUT_SIZE))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.gelu(x)
        x = self.layers[-1](x)
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
            if delta[0, i] != 0:
                losses += self._alpha * delta[0, i] + self._beta
        return losses
