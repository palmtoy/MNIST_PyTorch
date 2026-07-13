# 日期：2021年07月17日
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as tsf


batch_size = 64
transform = tsf.Compose([tsf.ToTensor(), tsf.Normalize([0.1307], [0.3081])])
# Normalize: 正则化, 降低模型复杂度, 防止过拟合

# 下载数据集: torchvision 已经预先实现了常用的 Dataset, 包括 MINST
train_set = datasets.MNIST(root = "data", train = True, download = True, transform = transform)
test_set = datasets.MNIST(root = "data", train = False, download = True, transform = transform)

# 加载数据集, 将数据集变成迭代器
def get_data_loader():
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)
    return train_loader, test_loader


# 显示数据集中的图片 ( matplotlib 网格布局, 一屏显示多张带标题 )
def show_images(rows = 8, cols = 8):
    import matplotlib.pyplot as plt

    # 单独建一个不做 Normalize 的数据集, 方便直接显示原图
    raw_set = datasets.MNIST(root = "data", train = True, download = True,
                             transform = tsf.ToTensor())
    plt.figure(figsize = (cols * 1.5, rows * 1.5))
    for i in range(rows * cols):
        image, label = raw_set[i]  # image 形状为 (1, 28, 28), 表示单通道灰度图
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image.squeeze().numpy(), cmap="gray")  # squeeze 去掉通道维
        plt.title(str(label))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_images()
