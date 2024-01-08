# APPLIED MACHINE LEARNING SYSTEM ELEC0134 22/23 REPORT

This is the project about image classification based on PneumoniaMNIST dataset and PathMNIST dataset. This is not the initial repository as the code was writte in Jupyter Notebook format, which could be found in https://github.com/samleong66/AMLS_final.git if needed. 

## Organization

- **/A**: Contains the code and deep learning model for task A.
- **/B**: Contains the code and deep learning model for task B.
- **/Datasets**: Contains the PneumoniaMNIST dataset for task A and the PathMNIST dataset for task B.

## Files

- **main.py**: Entry point for the application. There are few functions in the code and seperated into few blocks but all are annotated. If user need to run one of the block, just simply cancel the annotation. The description of every block is written above the corresponding code.
- **/A/task_A.py**: code for data pre-processing and models training of task A.
- **/B/task_B.py**: code for data pre-processing and models training of task B.
- **/Datasets/pathmnist.npz**: file contains PathMNIST dataset.
- **/Datasets/pneumoniamnist.npz**: file contains PneumoniaMNIST dataset
- **/A/resnet18.pth**: model file of ResNet18 for task A
- **/B/mlp.pth**: model file of MLP for task B
- **/B/resnet50.pth**: model file of ResNet50 for task B
- **requirements.txt**: List of packages and their versions required to run the code.


## Packages

Ensure you have the following packages installed before running the code:

```bash
pip install -r requirements.txt
