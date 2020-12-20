import numpy as np
import os
import cv2
import matplotlib.pyplot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import sklearn.metrics

# Parameters

np.random.seed(2)
random.seed(1)

resolution = 50
width = resolution
height = resolution
d_size = (width, height)
channels = 3
desired_size = (width, height, channels)

test_length_percent = 10

epochs = 15
learning_rate = 0.001
batch_size = 16
scheduler_step = 10
scheduler_gamma = 0.1

augmentation = True
zoom = True
zoom_pixel_limit = 3

# Load and label data

root = "./data"
class_names = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]

examples_count = []
images = []
labels = []
for class_index, name in enumerate(class_names):
    class_directory = os.path.join(root, name)
    files = os.listdir(class_directory)
    class_images = []
    class_labels = []
    examples_count.append(len(files))
    for file in files:
        if channels == 3:
            image = cv2.imread(os.path.join(class_directory, file))
        elif channels == 1:
            image = cv2.imread(os.path.join(class_directory, file), 0)
            image = np.reshape(image, newshape=(image.shape[0], image.shape[1], channels))
        elif channels == 4:
            image = cv2.imread(os.path.join(class_directory, file))
            image_grey = np.reshape(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                    newshape=(image.shape[0], image.shape[1], 1))
            img = np.concatenate((image, image_grey), axis=2)
            image = img
        class_images.append(image)
        class_labels.append(class_index)
    images.append(class_images)
    labels.append(class_labels)

total_examples = sum(examples_count)

# Display examples

examples_per_class = 2
class_count = len(class_names)
indexes_to_display = []
images_to_display = []

for i in range(class_count):
    ind_to_display = np.random.randint(0, len(images[i]), examples_per_class)
    indexes_to_display.append(ind_to_display)
    for index in ind_to_display:
        images_to_display.append(images[i][index])

fig_width = 12.0
fig_height = 6.0
figure_size = (fig_width, fig_height)
figure = plt.figure(figsize=figure_size)
figure.suptitle("Example pictures of each class")
cl = 0

for i in range(1, class_count * examples_per_class + 1):
    figure.add_subplot(examples_per_class, class_count, i)
    plt.axis("off")
    plt.title("Class: " + class_names[cl])
    if i % examples_per_class == 0:
        cl += 1
    plt.imshow(images_to_display[i - 1])


# plt.show()

# Resize images


def if_sizes_equal(desired_size, image_size_list):
    if_sizes_wrong = []
    for img_size in image_size_list:
        if img_size != desired_size:
            if_sizes_wrong.append(True)
        else:
            if_sizes_wrong.append(False)
    if any(if_sizes_wrong):
        print("At least one of {} example's size is wrong \n".format(len(image_size_list)))
        return False
    else:
        print("All of {} examples' sizes are correct \n".format(len(image_size_list)))
        return True


image_sizes_original = [img.shape for img in images_to_display]

for cl_index, class_array in enumerate(images):
    for img_index, image in enumerate(class_array):
        if image.shape != d_size:
            if channels == 3:
                images[cl_index][img_index] = cv2.resize(image, d_size)
            elif channels == 1 or channels == 4:
                images[cl_index][img_index] = np.reshape(cv2.resize(image, d_size), desired_size)

images_to_display_updated = []
for cl_index, indexes in enumerate(indexes_to_display):
    for index in indexes:
        images_to_display_updated.append(images[cl_index][index])

image_sizes_updated = [img.shape for img in images_to_display_updated]

print("Original examples' sizes:")
print(image_sizes_original)
if_sizes_equal(desired_size, image_sizes_original)

print("Updated examples' sizes:")
print(image_sizes_updated)
if_sizes_equal(desired_size, image_sizes_updated)

all_images_sizes = []
for class_images in images:
    for img in class_images:
        all_images_sizes.append(img.shape)

if if_sizes_equal(desired_size, all_images_sizes):
    print("All images have correct sizes \n")
else:
    print("At least one image to be scaled \n")


# Output details


def write_messages(messages, numbers, message_length=80):
    for message, number in zip(messages, numbers):
        row_length = len(message)
        message += "{:>" + str(message_length - row_length) + "}"
        print(message.format(number))


def output_details():
    global train_index_limits, messages, numbers
    examples_count, percentage_total, percentage_train, test_set_count = calculate_details()
    messages = ["Classes names: ",
                "Total examples in each class: ",
                "Training set images count in each class: ",
                "Test set images count in each class: ",
                "Percent of total images per each class:",
                "Percent of training images per each class:"]
    numbers = [str(class_names),
               str(examples_count),
               str(train_index_limits),
               str(test_set_count),
               str(percentage_total),
               str(percentage_train)]

    write_messages(messages, numbers)


def calculate_details():
    global train_index_limits
    examples_count = [len(i) for i in images]
    train_index_limits = [int(images_count * (100 - test_length_percent) / 100) for images_count in examples_count]
    test_set_count = [ex_count - tr_count for ex_count, tr_count in zip(examples_count, train_index_limits)]
    percentage_total = [round(examples / total_examples * 100, 1) for examples in examples_count]
    percentage_train = [round(tr_idx_lim / sum(train_index_limits) * 100, 1) for tr_idx_lim in train_index_limits]
    return examples_count, percentage_total, percentage_train, test_set_count


output_details()

# Shuffle data set

images_ordered = images

if_shuffle = True

if if_shuffle:
    for class_images in images:
        random.shuffle(class_images)

# Normalize data set

channel_mean = np.mean(np.concatenate(images), axis=(0, 1, 2))
channel_std_deviation = np.std(np.concatenate(images), axis=(0, 1, 2))

messages = ["Mean value of each channel: ",
            "Standard deviation of each channel: "]
numbers = [str(np.round(channel_mean, 2)),
           str(np.round(channel_std_deviation, 2))]

write_messages(messages, numbers)

for cl, class_images in enumerate(images):
    for i, image in enumerate(class_images):
        images[cl][i] = (image - channel_mean) / channel_std_deviation

channel_mean_normalized = np.mean(np.concatenate(images), axis=(0, 1, 2))
channel_std_deviation_normalized = np.std(np.concatenate(images), axis=(0, 1, 2))

messages = ["Mean value of each channel normalized: ",
            "Standard deviation of each channel normalized: "]

numbers = [str(np.round(channel_mean_normalized, 2)),
           str(np.round(channel_std_deviation_normalized, 2))]

write_messages(messages, numbers)

# Divide data into train and test sets

train_images = []
train_labels = []
test_images = []
test_labels = []

for i, class_images in enumerate(images):
    train_images.append(class_images[:train_index_limits[i]])
    train_labels.append(labels[i][:train_index_limits[i]])
    test_images.append(class_images[train_index_limits[i]:])
    test_labels.append(labels[i][train_index_limits[i]:])


# Augmentation

def augmentation(images_set, labels_set):
    training_examples_count = [len(c) for c in images_set]
    desired_examples_per_class = max(training_examples_count)
    missing_examples = [desired_examples_per_class - e_count for e_count in training_examples_count]

    picked_images = []
    for i, missing_count in enumerate(missing_examples):
        a = np.array(images_set[i])
        indices = np.random.choice(a.shape[0], missing_count)
        picked_images.append(np.take(a=a, indices=indices, axis=0))

    # Pseudo zoom
    if zoom:
        pixel_limit = zoom_pixel_limit

        for set, picked_set in enumerate(picked_images):
            for i, image in enumerate(picked_set):
                cut_low = np.random.randint(0, pixel_limit, size=2)
                cut_high = np.random.randint(resolution - pixel_limit, resolution, size=2)
                cut_image = image[cut_low[0]:cut_high[0], cut_low[1]:cut_high[1], :]
                zoomed_image = cv2.resize(cut_image, d_size)
                picked_images[set][i] = zoomed_image

    for i, picked in enumerate(picked_images):
        images_set[i].extend(picked)
        labels_set[i].extend([i for s in range(picked.shape[0])])

    return images_set, labels_set


if augmentation:
    train_images, train_labels = augmentation(train_images, train_labels)


# Data set class


class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# Loading data into data sets

train_images = torch.tensor(np.rollaxis(np.concatenate(train_images), axis=3, start=1), dtype=torch.float)
train_labels = torch.tensor(np.concatenate(train_labels), dtype=torch.long)
test_images = torch.tensor(np.rollaxis(np.concatenate(test_images), axis=3, start=1), dtype=torch.float)
test_labels = torch.tensor(np.concatenate(test_labels), dtype=torch.long)

train_dataset = DataSet(train_images, train_labels)
test_dataset = DataSet(test_images, test_labels)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=test_dataset.__len__())


# Network model class


class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        # Parameter

        # 1st - convolution layer
        input_channels_01 = channels
        output_channels_01 = 16
        filter_size_01 = 7
        padding_01 = self.same_padding(filter_size_01)
        stride_01 = 1
        max_pooling_kernel_01 = 2
        output_size_01 = self.get_output_size(input_size=input_size,
                                              filter_size=filter_size_01,
                                              padding=padding_01,
                                              stride=stride_01,
                                              pooling=max_pooling_kernel_01)

        # 2nd - convolution layer
        filter_size_02 = 5
        input_channels_02 = output_channels_01
        output_channels_02 = 24
        padding_02 = self.same_padding(filter_size_02)
        stride_02 = 1
        max_pooling_kernel_02 = 2
        output_size_02 = self.get_output_size(input_size=output_size_01,
                                              filter_size=filter_size_02,
                                              padding=padding_02,
                                              stride=stride_02,
                                              pooling=max_pooling_kernel_02)

        # 3rd - convolution layer
        filter_size_03 = 5
        input_channels_03 = output_channels_02
        output_channels_03 = 64
        padding_03 = self.same_padding(filter_size_03)
        stride_03 = 1
        self.max_pooling_kernel_03 = 1
        output_size_03 = self.get_output_size(input_size=output_size_02,
                                              filter_size=filter_size_03,
                                              padding=padding_03,
                                              stride=stride_03,
                                              pooling=self.max_pooling_kernel_03)

        # 4th - convolution layer
        filter_size_04 = 3
        input_channels_04 = output_channels_03
        output_channels_04 = 64
        padding_04 = self.same_padding(filter_size_04)
        stride_04 = 1
        self.max_pooling_kernel_04 = 2
        output_size_04 = self.get_output_size(input_size=output_size_03,
                                              filter_size=filter_size_04,
                                              padding=padding_04,
                                              stride=stride_04,
                                              pooling=self.max_pooling_kernel_04)

        # 5th - linear layer
        input_features_05 = int(pow(output_size_04, 2) * output_channels_04)
        output_features_05 = 1000
        dropout_p = 0.5

        # 6th - linear layer
        input_features_06 = output_features_05
        output_features_06 = 500

        # 7th - linear layer
        input_features_07 = output_features_06
        output_features_07 = class_count

        # Layers definitions

        self.cnn_01 = nn.Conv2d(in_channels=input_channels_01,
                                out_channels=output_channels_01,
                                kernel_size=filter_size_01,
                                stride=stride_01,
                                padding=padding_01)
        self.batch_normalization_01 = nn.BatchNorm2d(num_features=output_channels_01)
        self.max_pooling_01 = nn.MaxPool2d(kernel_size=max_pooling_kernel_01)

        self.cnn_02 = nn.Conv2d(in_channels=input_channels_02,
                                out_channels=output_channels_02,
                                kernel_size=filter_size_02,
                                stride=stride_02,
                                padding=padding_02)
        self.batch_normalization_02 = nn.BatchNorm2d(num_features=output_channels_02)
        self.max_pooling_02 = nn.MaxPool2d(kernel_size=max_pooling_kernel_02)

        self.cnn_03 = nn.Conv2d(in_channels=input_channels_03,
                                out_channels=output_channels_03,
                                kernel_size=filter_size_03,
                                stride=stride_03,
                                padding=padding_03)
        self.batch_normalization_03 = nn.BatchNorm2d(num_features=output_channels_03)
        self.max_pooling_03 = nn.MaxPool2d(kernel_size=self.max_pooling_kernel_03)

        self.cnn_04 = nn.Conv2d(in_channels=input_channels_04,
                                out_channels=output_channels_04,
                                kernel_size=filter_size_04,
                                stride=stride_04,
                                padding=padding_04)
        self.batch_normalization_04 = nn.BatchNorm2d(num_features=output_channels_04)
        self.max_pooling_04 = nn.MaxPool2d(kernel_size=self.max_pooling_kernel_04)

        self.linear_05 = nn.Linear(in_features=input_features_05, out_features=output_features_05)
        self.dropout = nn.Dropout(p=dropout_p)

        self.linear_06 = nn.Linear(in_features=input_features_06, out_features=output_features_06)

        self.linear_07 = nn.Linear(in_features=input_features_07, out_features=output_features_07)

        self.relu = nn.ReLU()

        messages = ["1st convolution layer filter size, input and output channels: ",
                    "2nd convolution layer filter size, input and output channels: ",
                    "3rd convolution layer filter size, input and output channels: ",
                    "4th convolution layer filter size, input and output channels: ",
                    "1st linear layer input and output features: ",
                    "2nd linear layer input and output features: ",
                    "3rd linear layer input and output features: "]
        numbers = [str(self.cnn_01.kernel_size) + str(", ") + str(self.cnn_01.in_channels) + str(", ") + str(
            self.cnn_01.out_channels),
                   str(self.cnn_02.kernel_size) + str(", ") + str(self.cnn_02.in_channels) + str(", ") + str(
                       self.cnn_02.out_channels),
                   str(self.cnn_03.kernel_size) + str(", ") + str(self.cnn_03.in_channels) + str(", ") + str(
                       self.cnn_03.out_channels),
                   str(self.cnn_04.kernel_size) + str(", ") + str(self.cnn_04.in_channels) + str(", ") + str(
                       self.cnn_04.out_channels),
                   str(self.linear_05.in_features) + str(", ") + str(self.linear_05.out_features),
                   str(self.linear_06.in_features) + str(", ") + str(self.linear_06.out_features),
                   str(self.linear_07.in_features) + str(", ") + str(self.linear_07.out_features)]
        write_messages(messages, numbers)

    def forward(self, x):
        out = self.cnn_01(x)
        out = self.batch_normalization_01(out)
        out = self.relu(out)
        out = self.max_pooling_01(out)

        out = self.cnn_02(out)
        out = self.batch_normalization_02(out)
        out = self.relu(out)
        out = self.max_pooling_02(out)

        out = self.cnn_03(out)
        out = self.batch_normalization_03(out)
        out = self.relu(out)
        if self.max_pooling_kernel_03 != 1:
            out = self.max_pooling_03(out)

        out = self.cnn_04(out)
        out = self.batch_normalization_04(out)
        out = self.relu(out)
        if self.max_pooling_kernel_04 != 1:
            out = self.max_pooling_04(out)

        out = out.view(-1, self.linear_05.in_features)

        out = self.linear_05(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear_06(out)
        out = self.relu(out)

        out = self.linear_07(out)
        return out

    def same_padding(self, filter):
        if filter is None:
            return 0
        else:
            return int((filter - 1) / 2)

    def get_output_size(self, input_size, filter_size, padding, stride, pooling):
        return int(((input_size - filter_size + 2 * padding) / stride + 1) / pooling)


# Define the model

model = CNN(resolution, class_count).float()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []


# Training


def train_model():
    for e in range(epochs):
        model.train()
        train_epoch_loss = 0
        batch_iterations = 0
        correct = 0
        for x, y in train_loader:
            y_hat = model(x)
            loss = loss_function(y_hat, y)
            train_epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, indices = torch.max(y_hat, 1)
            correct += (indices == y).sum().item()
            batch_iterations += 1

        scheduler.step()

        epoch_loss = round(train_epoch_loss / batch_iterations, 3)
        epoch_accuracy = round(correct / train_dataset.__len__(), 3)

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        messages = ["After epoch {} of {}, TRAIN set accuracy, loss: ".format(e + 1, epochs)]
        numbers = [str(epoch_accuracy) + str(",  ") + str(epoch_loss)]

        write_messages(messages, numbers)

        # Test phase
        test_epoch_loss = 0
        batch_iterations = 0
        correct = 0
        model.eval()
        batch_iterations, correct, test_epoch_loss, _ = test_model(loader=test_loader,
                                                                   batch_iterations=batch_iterations,
                                                                   correct=correct,
                                                                   test_epoch_loss=test_epoch_loss)

        epoch_loss = round(test_epoch_loss / batch_iterations, 3)
        epoch_accuracy = round(correct / test_dataset.__len__(), 3)

        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_accuracy)

        beginning_length = len("After epoch {} of {}, ")
        beginning_string = beginning_length * " "

        messages = [beginning_string + "TEST set accuracy, loss: ".format(e + 1, epochs)]
        numbers = [str(epoch_accuracy) + str(",  ") + str(epoch_loss)]

        write_messages(messages, numbers)


def test_model(loader, batch_iterations=0, correct=0, test_epoch_loss=0):
    for x, y in loader:
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        test_epoch_loss += loss.item()

        _, indices = torch.max(y_hat, 1)
        correct += (indices == y).sum().item()
        batch_iterations += 1
    return batch_iterations, correct, test_epoch_loss, indices


train_model()

# Results

plt_2 = matplotlib.pyplot
figure_2 = plt_2.figure(figsize=(10, 4))
figure_2.add_subplot(2, 1, 1)
plt_2.plot(train_loss, label="Train loss")
plt_2.plot(test_loss, label="Test loss")
figure_2.add_subplot(2, 2, 2)
plt_2.plot(train_accuracy, label="Train accuracy")
plt_2.plot(test_accuracy, label="Test accuracy")
figure_2.tight_layout()
plt_2.show()

# Metrics

_, _, _, test_set_indices = test_model(loader=test_loader)

confusion_matrix_test_set = sklearn.metrics.multilabel_confusion_matrix(test_labels, test_set_indices)
classification_report_test_set = sklearn.metrics.classification_report(test_labels, test_set_indices,
                                                                       target_names=class_names)

matrix_test_set = np.zeros((class_count, class_count))

for index, cl_predicted in enumerate(test_set_indices):
    matrix_test_set[test_labels[index], cl_predicted] += 1


def print_confusion_matrix(matrix):
    column_width = 10
    title = "Act\Pred->"
    line = title + " " * (column_width - len(title))
    for s1 in range(matrix.shape[0]):
        line += "{:>10}".format(class_names[s1])
    print(line)
    for s1 in range(matrix.shape[0]):
        line = "{:>10}".format(class_names[s1])
        for s2 in range(matrix.shape[1]):
            line += "{:>10}".format(int(matrix[s1, s2]))
        print(line)
    print("\n")


print_confusion_matrix(matrix_test_set)
print(classification_report_test_set)
