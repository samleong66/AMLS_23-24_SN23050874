from A import task_A
from B import task_B
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def show_augmentation_examples(data):
    aug = task_B.Augment()
    fig = plt.figure(figsize=(10, 15))
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(aug.invert(data))
    ax.set_title('Invert')
    ax = fig.add_subplot(3, 2, 2)
    ax.imshow(aug.blur(data))
    ax.set_title('Blur')
    ax = fig.add_subplot(3, 2, 3)
    ax.imshow(aug.sq_blur(data))
    ax.set_title('Partly Blur')
    ax = fig.add_subplot(3, 2, 4)
    ax.imshow(aug.random_brightness(data))
    ax.set_title('Random Brightness')
    ax = fig.add_subplot(3, 2, 5)
    ax.imshow(aug.rotate(data))
    ax.set_title('Rotate')
    ax = fig.add_subplot(3, 2, 6)
    ax.imshow(aug.gray_scale(data))
    ax.set_title('Gray')

if __name__ == '__main__':
################################################################
#                  This section is for Task A                  #
################################################################
    '''
    This code is to show example of PneumoniaMNIST dataset
    '''
    # raw_data = task_A.load_Pneumoniamnist('raw')
    # task_A.show_example(raw_data)
    '''
    This code is to search best parameter set and train for kNN
    '''
    # training_data, testing_data = task_A.load_Pneumoniamnist('with_augmentation')
    # grid_search = task_A.search_parameters('kNN', training_data, testing_data)
    # clf = task_A.train_model('kNN', training_data, testing_data, grid_search)
    '''
    This code is to search best parameter set and train for kNN
    '''
    # training_data, testing_data = task_A.load_Pneumoniamnist('with_augmentation')
    # grid_search = task_A.search_parameters('kNN', training_data, testing_data)
    # clf = task_A.train_model('kNN', training_data, testing_data, grid_search)
    '''
    This code is to search best parameter set and train for SVM
    '''
    # training_data, testing_data = task_A.load_Pneumoniamnist('with_augmentation')
    # grid_search = task_A.search_parameters('SVM', training_data, testing_data)
    # clf = task_A.train_model('SVM', training_data, testing_data, grid_search)
    '''
    This code is to search best parameter set and train for MLP
    '''
    # training_data, testing_data = task_A.load_Pneumoniamnist('with_augmentation')
    # grid_search = task_A.search_parameters('MLP', training_data, testing_data)
    # clf = task_A.train_model('MLP', training_data, testing_data, grid_search)
    '''
    This code is to train a ResNet18 model
    '''
    # training_data, testing_data = task_A.load_Pneumoniamnist('with_augmentation', False, False)
    # model = task_A.train_model('ResNet18', training_data, testing_data)
    # torch.save({'model_state_dict': model.state_dict()}, 'A/resnet18.pth')
    '''
    This code is to read a ResNet18 model and print accuracy in test dataset
    '''
    # training_data, testing_data = task_A.load_Pneumoniamnist('with_augmentation', False, False)
    # model = task_A.resnet18(2, True)
    # checkpoint = torch.load('A/resnet18.pth')
    # data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    # PneumoniaMNIST_test = task_A.get_dataset(testing_data[0], testing_data[1], transforms=data_transform)
    # PneumoniaMNIST_test = DataLoader(PneumoniaMNIST_test, batch_size=16, shuffle=True)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to('cuda')
    # print(task_A.compute_accuracy(model, PneumoniaMNIST_test, device='cuda').item())
################################################################
#                  This section is for Task B                  #
################################################################
    '''
    This code is to show example of PathMNIST dataset
    '''
    # raw_data = task_B.load_Pathmnist('raw')
    # task_B.show_example(raw_data)
    '''
    This code is to search best parameter set and train for kNN
    '''
    # training_data, testing_data = task_B.load_Pathmnist('with_augmentation')
    # grid_search = task_B.search_parameters('kNN', training_data)
    # clf = task_B.train_model('kNN', training_data, testing_data, grid_search)
    '''
    This code is to search best parameter set and train for SVM
    '''
    # training_data, testing_data = task_B.load_Pathmnist('with_augmentation')
    # grid_search = task_B.search_parameters('SVM', training_data)
    # clf = task_B.train_model('SVM', training_data, testing_data, grid_search)
    '''
    This code is to train a MLP model
    '''
    # training_data, testing_data = task_B.load_Pathmnist('with_augmentation', False, False)
    # model = task_B.train_model('MLP', training_data, testing_data)
    # torch.save({'model_state_dict': model.state_dict()}, 'B/mlp.pth')
    '''
    This code is to read a MLP model and print accuracy in test dataset
    '''
    # training_data, testing_data = task_B.load_Pathmnist('with_augmentation', False, False)
    # model = task_B.MLP(input_num=28 * 28 * 3, output_num=9, hidden_layers=(50, 50))
    # checkpoint = torch.load('B/MLP.pth')
    # data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    # PathMNIST_test = task_B.get_dataset(testing_data[0], testing_data[1], transforms=data_transform)
    # PathMNIST_test = DataLoader(PathMNIST_test, batch_size=128, shuffle=True)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to('cuda')
    # print(task_B.test(model, PathMNIST_test).item() * 100 / 7180)
    '''
    This code is to train a ResNet50 model
    '''
    # training_data, testing_data = task_B.load_Pathmnist('with_augmentation', False, False)
    # model = task_B.train_model('ResNet50', training_data, testing_data)
    # torch.save({'model_state_dict': model.state_dict()}, 'B/resnet50.pth')
    '''
    This code is to read a ResNet18 model
    '''
    # training_data, testing_data = task_B.load_Pathmnist('with_augmentation', False, False)
    # model = task_B.resnet50(9, False)
    # checkpoint = torch.load('B/resnet50.pth')
    # data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    # PathMNIST_test = task_B.get_dataset(testing_data[0], testing_data[1], transforms=data_transform)
    # PathMNIST_test = DataLoader(PathMNIST_test, batch_size=128, shuffle=True)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to('cuda')
    # print(task_B.compute_accuracy(model, PathMNIST_test, device='cuda').item())
################################################################
#       This section is to show images after augmented         #
################################################################
    # raw_data = task_B.load_Pathmnist('raw')
    # show_augmentation_examples(raw_data['train_images'][0])
    plt.show()
