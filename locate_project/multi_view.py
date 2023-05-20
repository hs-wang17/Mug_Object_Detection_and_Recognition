import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
from tqdm import tqdm, trange
import pdb
import sys

x_list = []
y_list = []
for pic in tqdm(os.listdir('scene_all')):
    pic_path = './scene_all/' + pic
    # 读取RGB三通道图像(640, 640, 3)
    pic_data = cv2.imread(pic_path, cv2.IMREAD_COLOR)
    pic_data = cv2.resize(pic_data, (640, 640))
    x_list.append(pic_data)
    y_list.append(int(pic[6:8]))# scene序号
x_list = np.array(x_list)  # scene: (2500, 640, 1137, 3), object: (15000, 640, 640, 3)
y_list_int = np.array(y_list)

scene_label = pd.read_excel('scene_label.xlsx')
y_list = np.zeros((y_list_int.shape[0], 20))
for i in trange(y_list_int.shape[0]):
    y_list[i, scene_label[scene_label.id==y_list_int[i]].iloc[:, 1:].dropna(axis=1).astype(int).to_numpy()[0].tolist()] = 1

label_list = []
for lab in tqdm(os.listdir('scene_label_all')):
    lab_path = './scene_label_all/' + lab
    label_list_local = [[0] * 5 for _ in range(19)]  # 初始化一个19*5的全0列表
    with open(lab_path, 'r') as f:  # 请将'file.txt'替换为你的txt文件的真实路径
        lines = f.readlines()
        for line in lines:
            data = line.split()
            category = int(data[0])  # 类别
            values = [float(x) for x in data[1:]]  # 对应的数据
            label_list_local[category] = [category] + values  # 更新对应类别的数据
    label_list.append(label_list_local)
label_list = np.array(label_list).astype('float32')


from sklearn.model_selection import StratifiedShuffleSplit


# 假设标签数据保存在label_list中，其中每个标签是一个整数
X = x_list
y = label_list

# 分层抽样，其中train_size和test_size分别表示训练集和测试集的比例
# n_splits表示抽取的次数，random_state表示随机数种子
split = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=42)
train_index, test_index = next(split.split(X, y_list))

# 得到训练集和测试集
X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
y_list_train, y_list_test = [y_list[i] for i in train_index], [y_list[i] for i in test_index]


# 将训练集进一步划分为训练集和验证集，其中test_size表示验证集的比例
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, valid_index = next(split.split(X_train, y_list_train))


# 得到训练集、验证集和测试集的索引
train_index = [train_index[i] for i in range(len(train_index))]
valid_index = [valid_index[i] for i in range(len(valid_index))]
test_index = [test_index[i] for i in range(len(test_index))]


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
# 创建一个SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# 定义超参数
batch_size = 8
learning_rate = 0.0001
num_epochs = 1000

# 设置 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data) // 2

    def __getitem__(self, index):
        img = self.data[index]
        img = torch.cat((self.transform(self.data[index * 2]), self.transform(self.data[index * 2 + 1])), dim=2)  # 假设图片在宽度方向上拼接
        label = torch.cat((torch.tensor(self.labels[index*2]), torch.tensor(self.labels[index*2+1])), dim=0)  # 标签在第一维度上拼接
        # if self.transform is not None:
        #     img = self.transform(img)
        return img, label

class MyTestDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data) // 2

    def __getitem__(self, index):
        img = self.data[index]
        img = torch.cat((self.transform(self.data[index * 2]), self.transform(self.data[index * 2 + 1])), dim=2)  # 假设图片在宽度方向上拼接
        label = torch.cat((torch.tensor(self.labels[index*2]), torch.tensor(self.labels[index*2+1])), dim=0)  # 标签在第一维度上拼接
        # if self.transform is not None:
        #     img = self.transform(img)
        img1 = self.data[index * 2]
        img2 = self.data[index * 2 + 1]
        return img, label, img1, img2

# 加载数据集
train_data = x_list[train_index]
train_labels = y[train_index]
valid_data = x_list[valid_index]
valid_labels = y[valid_index]
test_data = x_list[test_index]
test_labels = y[test_index]

# 定义数据增强和标准化
# 在scene数据集中只做了标准化
transform = transforms.Compose([
#     transforms.RandomAffine(5),
#     transforms.ColorJitter(hue=.05, saturation=.05),
#     transforms.RandomCrop((88, 88)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化张量
])

# 加载数据集
train_dataset = MyDataset(train_data, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = MyDataset(valid_data, valid_labels, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MyTestDataset(train_data, train_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 定义模型
class MyModel(nn.Module):
    def __init__(self, num_classes=19):
        super(MyModel, self).__init__()
        self.img_num = 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 20 * 20 * self.img_num, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes * self.img_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 20 * 20 * self.img_num)# 最后一维表示图片个数
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

total_class = 19
model = MyModel(num_classes=total_class * 5)
model.to(device)

# 定义损失函数和优化器
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_iou(box1, box2):
    # box = [x_center, y_center, width, height]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate the intersection coordinates
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Calculate Intersection Area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate Union Area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = box1_area + box2_area - intersection

    # Avoid divide by zero
    epsilon = 1e-5

    # Calculate IoU
    iou = intersection / (union + epsilon)
    return iou

EVAL = True
if EVAL == True:
    epoch = 6344
    model = MyModel(num_classes=total_class * 5)
    model.load_state_dict(torch.load('./classifier/model-CNN-scene/epoch-%d.pt' % epoch))
    model.to(device)
    IoU_total = 0
    count = 0
    for inputs, labels, img1, img2 in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batchsize_local = labels.shape[0]
        outputs = model(inputs)
        output_classification = outputs[:, :total_class * 2]
        output_regression = outputs[:, total_class * 2:]
        labels_classification = labels[:, :, 0]  # CrossEntropyLoss期望的标签类型为long
        labels_regression = labels[:, :, 1:].reshape(batchsize_local, -1)  # 调整形状以匹配output_regression
        # print('label:', labels_regression)
        # print('output:', output_regression)
        # plot and save, the first dimension is batchsize
        for i in range(batchsize_local):
            image1 = img1[i].numpy()
            image2 = img2[i].numpy()
            # Resize the image
            image1 = Image.fromarray((image1).astype(np.uint8))
            image1 = image1.resize((1137, 640))
            image2 = Image.fromarray((image2).astype(np.uint8))
            image2 = image2.resize((1137, 640))
            fig, ax = plt.subplots(1)
            ax.imshow(image1)
            box = output_regression[i].cpu().reshape(total_class * 2, 4)
            box_real = labels_regression[i].cpu().reshape(total_class * 2, 4)
            for j in [1, 4, 6]:
                box_item = box[j].detach().numpy()
                box_real_item = box_real[j].detach().numpy()
                iou = calculate_iou(box_item, box_real_item)
                IoU_total += iou
                count += 1
                # change the box format from yolo format to matplotlib format
                box_item = [(box_item[0] - box_item[2] / 2) * 1137, (box_item[1] - box_item[3] / 2) * 640,
                            box_item[2] * 1137, box_item[3] * 640]
                box_real_item = [(box_real_item[0] - box_real_item[2] / 2) * 1137,
                                 (box_real_item[1] - box_real_item[3] / 2) * 640, box_real_item[2] * 1137,
                                 box_real_item[3] * 640]
                # draw the box
                rect = patches.Rectangle((box_item[0], box_item[1]), box_item[2], box_item[3], linewidth=1,
                                         edgecolor='r', facecolor='none')
                rect_real = patches.Rectangle((box_real_item[0], box_real_item[1]), box_real_item[2], box_real_item[3],
                                              linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.add_patch(rect_real)
                # save the image
            # plt.savefig('image_with_boxes_%d.png' % i)
            # plt.show()
            plt.close()
    for inputs, labels, img1, img2 in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batchsize_local = labels.shape[0]
        outputs = model(inputs)
        output_classification = outputs[:, :total_class * 2]
        output_regression = outputs[:, total_class * 2:]
        labels_classification = labels[:, :, 0]  # CrossEntropyLoss期望的标签类型为long
        labels_regression = labels[:, :, 1:].reshape(batchsize_local, -1)  # 调整形状以匹配output_regression
        # print('label:', labels_regression)
        # print('output:', output_regression)
        # plot and save, the first dimension is batchsize
        for i in range(batchsize_local):
            image1 = img1[i].numpy()
            image2 = img2[i].numpy()
            # Resize the image
            image1 = Image.fromarray((image1).astype(np.uint8))
            image1 = image1.resize((1137, 640))
            image2 = Image.fromarray((image2).astype(np.uint8))
            image2 = image2.resize((1137, 640))
            fig, ax = plt.subplots(1)
            ax.imshow(image2)
            box = output_regression[i].cpu().reshape(total_class * 2, 4)
            box_real = labels_regression[i].cpu().reshape(total_class * 2, 4)
            for j in [1+total_class, 4+total_class, 6+total_class]:
                box_item = box[j].detach().numpy()
                box_real_item = box_real[j].detach().numpy()
                iou = calculate_iou(box_item, box_real_item)
                IoU_total += iou
                count += 1
                # change the box format from yolo format to matplotlib format
                box_item = [(box_item[0] - box_item[2] / 2) * 1137, (box_item[1] - box_item[3] / 2) * 640,
                            box_item[2] * 1137, box_item[3] * 640]
                box_real_item = [(box_real_item[0] - box_real_item[2] / 2) * 1137,
                                 (box_real_item[1] - box_real_item[3] / 2) * 640, box_real_item[2] * 1137,
                                 box_real_item[3] * 640]
                # draw the box
                rect = patches.Rectangle((box_item[0], box_item[1]), box_item[2], box_item[3], linewidth=1,
                                         edgecolor='r', facecolor='none')
                rect_real = patches.Rectangle((box_real_item[0], box_real_item[1]), box_real_item[2], box_real_item[3],
                                              linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.add_patch(rect_real)
                # save the image
            # plt.savefig('image_with_boxes_%d.png' % i)
            plt.close()

    print(IoU_total/count)
    sys.exit()

# 训练模型
min_losss_val = 100
for epoch in trange(num_epochs):
    running_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batchsize_local = labels.shape[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        output_classification = outputs[:, :total_class*2]
        output_regression = outputs[:, total_class*2:]
        labels_classification = labels[:, :, 0]  # CrossEntropyLoss期望的标签类型为long
        labels_regression = labels[:, :, 1:].reshape(batchsize_local, -1)  # 调整形状以匹配output_regression

        loss1 = criterion_1(output_classification, labels_classification)
        loss2 = criterion_2(output_regression, labels_regression)
        loss = loss2*100

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    # 假设你已经获取了loss和accuracy
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, epoch_loss))
    if not epoch % 10 :
        torch.save(model.state_dict(), "./classifier/model-CNN-scene/epoch-%d.pt" % epoch)

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            batchsize_local = labels.shape[0]
            outputs = model(inputs)

            output_classification = outputs[:, :total_class * 2]
            output_regression = outputs[:, total_class * 2:]
            labels_classification = labels[:, :, 0]  # CrossEntropyLoss期望的标签类型为long
            labels_regression = labels[:, :, 1:].reshape(batchsize_local, -1)  # 调整形状以匹配output_regression

            loss1 = criterion_1(output_classification, labels_classification)
            loss2 = criterion_2(output_regression, labels_regression)
            loss = loss2*100

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(valid_dataset)
        print('Loss of the model on the valid images: %f' % epoch_loss)
        if epoch_loss < min_losss_val:
            min_losss_val = epoch_loss
            torch.save(model.state_dict(), "./classifier/model-CNN-scene/epoch-6344.pt")
            print('Best model!!!!!!!! : %d' % epoch)

