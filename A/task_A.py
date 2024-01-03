import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import svm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader


class Augment():
    def __init__(self):
        self.inv_prob = 0.5
        self.blur_prob = 0.3
        self.sq_blur_prob = 0.3
        self.bright_prob = 0.5
        self.rotate_prob = 1.
        self.gray_prob = 0.0

    def invert(self, image):
        return 255 - image

    def blur(self, image):
        return cv2.blur(image, (3, 3))

    def sq_blur(self, image):
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        return image

    def random_brightness(self, image):
        c = random.uniform(0.2, 1.8)
        blank = np.zeros(image.shape, image.dtype)
        dst = cv2.addWeighted(image, c, blank, 1 - c, 0)
        return dst

    def rotate(self, image, scale=1.0):
        angle = random.uniform(-5, 5)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def gray_scale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.merge((gray, gray, gray))
        return dst

    def apply(self, image):
        inv_prob = random.random()
        blur_prob = random.random()
        sq_blur_prob = random.random()
        bright_prob = random.random()
        rotate_prob = random.random()

        if inv_prob < self.inv_prob:
            image = self.invert(image)

        if bright_prob < self.bright_prob:
            image = self.random_brightness(image)

        if rotate_prob < self.rotate_prob:
            image = self.rotate(image)

        if blur_prob < self.blur_prob:
            image = self.blur(image)

        if sq_blur_prob < self.sq_blur_prob:
            image = self.sq_blur(image)

        return image


def load_Pneumoniamnist(type: str, normalization=True, flatten=True):
    ac = np.load('Datasets/pneumoniamnist.npz')
    if type == 'raw':
        return ac

    x_train, y_train = ac['train_images'], ac['train_labels']
    x_val, y_val = ac['val_images'], ac['val_labels']
    x_test, y_test = ac['test_images'], ac['test_labels']
    x_train, y_train = np.vstack((x_train, x_val)), np.vstack((y_train, y_val))
    if type == 'no_augmentation':
        if normalization:
            x_train, x_test = x_train / 255, x_test / 255
        if flatten:
            x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)
            y_train, y_test = y_train.ravel(), y_test.ravel()
        return (x_train, y_train), (x_test, y_test)

    x_train_aug = []
    y_train_aug = []
    aug = Augment()
    for i in range(1000):
        idx = random.randint(0, len(x_train) - 1)
        fig = x_train[idx]
        y = y_train[idx]
        fig_t = aug.apply(fig)
        x_train_aug.append(fig_t)
        y_train_aug.append(y)
    x_train_aug, y_train_aug = np.array(x_train_aug), np.array(y_train_aug)
    x_train_aug, y_train_aug = np.vstack((x_train, x_train_aug)), np.vstack((y_train, y_train_aug))
    if type == 'with_augmentation':
        if normalization:
            x_train_aug, x_test = x_train_aug/255, x_test/255
        if flatten:
            x_train_aug, x_test = x_train_aug.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)
            y_train_aug, y_test = y_train_aug.ravel(), y_test.ravel()
        return (x_train_aug, y_train_aug), (x_test, y_test)


def show_example(raw_data):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(raw_data['train_images'][0], cmap='gray')
    ax.set_title('Pneumonia')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(raw_data['train_images'][5], cmap='gray')
    ax.set_title('Normal')
    # plt.show()


def search_parameters(model_name, training_data, testing_data):
    x_train, y_train = training_data[0], training_data[1]
    x_test, y_test = testing_data[0], testing_data[1]
    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    model, param_grid = None, None
    if model_name == 'kNN':
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15]}
        model = KNeighborsClassifier()
    elif model_name == 'SVM':
        param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.1, 1, 10, 100]},
                      {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
        model = svm.SVC()
    elif model_name == 'MLP':
        param_grid = {
            'hidden_layer_sizes': [(10, 10,), (50, 50,), (100, 100,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001],
        }
        model = MLPClassifier(max_iter=1000)
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    results = grid_search.cv_results_
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_test_score = results['mean_test_score'][i]
        if model_name == 'kNN':
            model = KNeighborsClassifier(**param_set)
        elif model_name == 'SVM':
            model = svm.SVC(**param_set)
        elif model_name == 'MLP':
            model = MLPClassifier(**param_set, max_iter=1000)
        model.fit(x_train, y_train)
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{param_set}\nAUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves for Different Hyperparameter Combinations')
    plt.legend(loc='lower right')
    # plt.show()
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Best Accuracy: {grid_search.best_score_}')
    return grid_search


class get_dataset():
    def __init__(self, data, labels: np.array, transforms=None):
        self.data = data
        self.labels = labels.ravel()
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, label


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def resnet18(num_classes, grayscale):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


def train_model(model_name, training_data, testing_data, grid_search=None):
    x_train, y_train = training_data[0], training_data[1]
    x_test, y_test = testing_data[0], testing_data[1]
    clf = None
    to =time()
    if model_name == 'kNN':
        clf = KNeighborsClassifier(**grid_search.best_params_).fit(x_train, y_train)
    elif model_name == 'SVM':
        clf = svm.SVC(**grid_search.best_params_).fit(x_train, y_train)
    elif model_name == 'MLP':
        clf = MLPClassifier(**grid_search.best_params_, max_iter=1000).fit(x_train, y_train)
    elif model_name == 'ResNet18':
        NUM_EPOCHS = 50
        BATCH_SIZE = 16
        NUM_CLASSES = 2
        GRAYSCALE = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
        PneumoniaMNIST_train = get_dataset(x_train, y_train, transforms=data_transform)
        PneumoniaMNIST_train = DataLoader(PneumoniaMNIST_train, batch_size=BATCH_SIZE, shuffle=True)
        PneumoniaMNIST_test = get_dataset(x_test, y_test, transforms=data_transform)
        PneumoniaMNIST_test = DataLoader(PneumoniaMNIST_test, batch_size=BATCH_SIZE, shuffle=True)

        model = resnet18(NUM_CLASSES, GRAYSCALE)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        avg_cost = []
        train_acc = []
        test_acc = []

        for epoch in range(NUM_EPOCHS):

            model.train()
            c = 0
            for batch_idx, (features, targets) in enumerate(PneumoniaMNIST_train):

                features = features.to(device)
                targets = targets.to(device)

                ### FORWARD AND BACK PROP
                logits, probas = model(features)
                cost = F.cross_entropy(logits, targets)
                optimizer.zero_grad()

                cost.backward()

                ### UPDATE MODEL PARAMETERS
                optimizer.step()

                ### LOGGING
                if not batch_idx % 50:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                          % (epoch + 1, NUM_EPOCHS, batch_idx,
                             len(PneumoniaMNIST_train), cost))
                c += cost
            avg_cost.append(c.item() / len(PneumoniaMNIST_train))

            model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference
                acc = compute_accuracy(model, PneumoniaMNIST_train, device=device)
                train_acc.append(acc.item())
                print('Epoch: %03d/%03d | Train: %.3f%%' % (epoch + 1, NUM_EPOCHS, acc))
                acc = compute_accuracy(model, PneumoniaMNIST_test, device=device)
                test_acc.append(acc.item())
                print('Epoch: %03d/%03d | Test: %.3f%%' % (epoch + 1, NUM_EPOCHS, acc))
                print('----------------------------------')

        print('Training time %d s'%(time() - to))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(avg_cost)), avg_cost)
        ax.set_title('Loss of each epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(train_acc)), train_acc)
        ax.set_title('Accuracy in training dataset of each epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(test_acc)), test_acc)
        ax.set_title('Accuracy in testing dataset of each epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.grid(True)

        return model

    print('Training time: {:.5f}s'.format(time() - to))
    predictions_labels = clf.predict(x_test)
    print('\r\nEvaluation:')
    print((classification_report(y_test, predictions_labels)))
    return clf