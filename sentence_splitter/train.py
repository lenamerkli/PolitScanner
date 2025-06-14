import sys
sys.path.append('/home/lena/Documents/python/PolitScanner/util')

from sentence_splitter import SentenceSplitter, INPUT_SIZE
from data import SentenceSplitterDataset
from util.time import current_time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_norm_
from torch.nn import Linear, HuberLoss
from torch import save, isnan
from pathlib import Path
from json import dump as json_dump


ALPHA = 1.1
BETA = 1.1
NUM_EPOCHS = 128
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
MIN_LENGTH = 512
GRADIENT_CLIP = 0.5


def required_loss(loss_history: list[float]) -> float:
    array = sorted(loss_history[-(NUM_EPOCHS // 8):])
    if len(array) >= 2:
        return array[1]
    return array[0]


def main() -> None:
    global NUM_EPOCHS
    start_time = current_time()
    model = SentenceSplitter().to('cuda')
    for layer in model.children():
        if isinstance(layer, Linear):
            xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()
    criterion = HuberLoss(delta=2.0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-6)
    train_loader = DataLoader(SentenceSplitterDataset(min_length=MIN_LENGTH, max_length=INPUT_SIZE), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    n_total_steps = len(train_loader)
    i = 0
    loss_history = []
    keyboard_interrupt = False
    train = True
    epoch = 0
    mean_loss = 0
    while train:
        try:
            if keyboard_interrupt:
                raise KeyboardInterrupt()
            model.train()
            for i, values in enumerate(train_loader):
                inputs = values[0]
                outputs = values[1]
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, outputs)
                if loss.dim() > 0:
                    mean_loss = loss.mean()
                else:
                    mean_loss = loss
                mean_loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
                optimizer.step()
                loss_history.append(mean_loss.item())
                if i % 100 == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None and isnan(param.grad).any():
                            print(f"NaN gradient in {name}")
                        if isnan(param).any():
                            print(f"NaN parameter in {name}")
                    print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i + 1}/{n_total_steps}], Loss: {loss_history[-1]:.6f}, Avg Loss: {sum(loss_history) / len(loss_history):.6f}")
                if i >= 4096:
                    break
            epoch += 1
            scheduler.step(mean_loss)
        except KeyboardInterrupt:
            if not keyboard_interrupt:
                print('Keyboard interrupt detected')
            train = False
            NUM_EPOCHS = epoch
        try:
            if loss_history:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{i + 1}/{n_total_steps}], Loss: {loss_history[-1]:.6f}")
            if epoch >= NUM_EPOCHS and loss_history[-1] <= required_loss(loss_history):
                train = False
        except KeyboardInterrupt:
            print('Keyboard interrupt detected')
            keyboard_interrupt = True
    print('Finished Training')
    end_time = current_time()
    save(model.state_dict(), Path(f"./models/{end_time}.pt").absolute())
    model_data = {
        'alpha': ALPHA,
        'beta': BETA,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'min_length': MIN_LENGTH,
        'start_time': start_time,
        'end_time': end_time,
        'loss_history': loss_history,
        'loss_ao10': sum(loss_history[-10:-1]) / 10.0 if len(loss_history) >= 10 else None,
    }
    with open(Path(f"./models/{end_time}.json").absolute(), 'w') as f:
        json_dump(model_data, f, indent=4)
    print(f"Model saved as {end_time}.pt")


if __name__ == '__main__':
    main()
