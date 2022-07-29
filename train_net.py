import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import net
import func
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = PIL.Image.open(img_path)
        label = 0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
      X = X.to(device)
      output = model(X)
      train_loss = loss_fn(output, X)
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
      train_loss_all.append(train_loss.item())
      train_loss_mse.append(loss_fn.mse.item())
      train_loss_ssim.append(loss_fn.ssim.item())
      if batch % 80 == 0:
        loss, current = train_loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}] {(current/size*100):>0.1f}%")


def test(dataloader, model, loss_fn):
    batch_size = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
          X = X.to(device)
          pred = model(X)
          test_loss += loss_fn(pred, X).item()
    test_loss /= batch_size
    test_loss_all.append(test_loss)
    print(f"Avg loss: {test_loss:>8f}")


setup_seed(40)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# ------------------load cifar10 & tno dataset----------------------
train_data_cifar10 = torchvision.datasets.CIFAR10(
    root='data/cifar-10-python',
    train=True,
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor(),
                                  ]),
    download=False
)
train_data_tno = CustomImageDataset(
    img_dir='TNO_copy',
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.CenterCrop(100),
                                  transforms.Resize([32, 32]),
                                  transforms.ToTensor(),
                                  ]),)
train_data = train_data_cifar10+train_data_tno
test_data = torchvision.datasets.CIFAR10(
    root='data/cifar-10-python',
    train=False,
    transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.ToTensor(),
                                  # transforms.Normalize(0.5, 0.5)
                                  ]),
    download=False
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=16,    # 指定batch大小
    shuffle=True,    # 每次迭代是否打乱顺序
    num_workers=0    # 进程数
)


fusionNet = net.DenseFuse()
optimizer = torch.optim.Adam(fusionNet.parameters(), lr=0.0001)
loss_fn = func.LossFunc(lam=1, window_size=5)
train_loss_all = []
train_loss_mse = []
train_loss_ssim = []
test_loss_all = []

for epoch in range(4):
    print(f'-----------epoch:{epoch:>3}-----------')
    train(train_loader, fusionNet, loss_fn, optimizer)

plt.figure()
plt.plot(train_loss_all[:5000], "r-")
plt.title("Train loss per iteration")
plt.figure()
plt.plot(train_loss_mse[:5000], "r-")
plt.title("Train mse_loss per iteration")
plt.figure()
plt.plot(train_loss_ssim[:5000], "r-")
plt.title("Train ssim_loss per iteration")
plt.show()
torch.save(fusionNet.state_dict(), 'model_weights_mixed_ssim_mse.pth')
func.save_object('loss_all.pickle', train_loss_all)
func.save_object('loss_mse.pickle', train_loss_mse)
func.save_object('loss_ssim.pickle', train_loss_ssim)
func.save_object('loss_test.pickle', test_loss_all)