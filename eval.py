# 日期：2021年07月17日
import torch
from dataset import get_data_loader


if __name__ == "__main__":
    _, eval_loader = get_data_loader()  # 因为没有验证集, 所以将测试集作为验证集使用。
    batch_size = 64
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (M1/M2/M3) GPU 加速
    else:
        device = torch.device("cpu")
    model = torch.load("save_model/model.pt", weights_only = False).to(device)  # 加载模型并搬到对应设备
    model.eval()  # 设置为验证模式

    acc = 0.
    with torch.no_grad():
        for digit, label in eval_loader:
            digit, label = digit.to(device), label.to(device)
            output = model(digit)  # 模型输出
            predict = output.max(dim=1, keepdim=True)[1]
            # 找到概率最大值的下标, 1表示按行计算。
            # max()返回两个值, 第一个是值, 第二个是索引, 所以取 max[1]

            acc += predict.eq(label.view_as(predict)).sum().item()
        accuracy = acc/len(eval_loader.dataset) * 100
        print("eval accuracy: {: .4f}%".format(accuracy))
