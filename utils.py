import os
import torch
import pickle


def setup_directories(directories):
    """Create required directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def initialize_device():
    """Initialize CUDA device if available, else CPU."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    return device


def prepare_dataset(dataset_cls, download_dir, train=True, transform=None, max_samples=None):
    """
    Prepare a subset of a dataset.

    Args:
        dataset_cls: torchvision dataset class (e.g., CIFAR100)
        download_dir: directory to save dataset
        train: bool, whether to load training or test data
        transform: transformations to apply on the dataset
        max_samples: maximum number of samples to extract

    Returns:
        images (Tensor): stacked image tensors
        targets (Tensor): corresponding labels
    """
    dataset = dataset_cls(download_dir, train=train,
                          transform=transform, download=True)
    images, targets = zip(*list(dataset)[:max_samples])
    return torch.stack(images), torch.tensor(targets)


def save_model(epoch, model, optimizer, loss, model_path):
    """Save the model state and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(model_path, 'model.pth'))


def load_model(model, model_path, device):
    """Load a model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_results(results, path):
    """Save results (dictionary) as a pickle file."""
    with open(f"{path}.pkl", "wb") as file:
        pickle.dump(results, file)
