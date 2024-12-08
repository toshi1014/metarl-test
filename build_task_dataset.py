import torch
import numpy as np
import random
from torch.utils.data import Dataset


# Set a random seed for reproducibility
def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


# Select a specified number of random items from a list
def select(items, num_samples):
    return random.sample(items.tolist(), num_samples)


# Create a batch of tasks from a taskset
def create_batch_of_tasks(taskset, is_shuffle=True, outer_batch_size=4):
    indices = list(range(len(taskset)))
    if is_shuffle:
        random.shuffle(indices)
    selected_indices = select(torch.tensor(indices), outer_batch_size)
    return [taskset[i] for i in selected_indices]


# Build task datasets for meta-learning
def build_task_dataset(img, target, num_all_task, num_task, k_support, k_query, num_class, inner_batch, is_val=False):
    # Shuffle image and label data
    shuffled_indices = torch.randperm(len(img))
    img, target = img[shuffled_indices], target[shuffled_indices]

    support_imgs, support_targets = [], []
    query_imgs, query_targets = [], []

    for task_idx in range(num_task):
        task_base = torch.randint(
            0, num_all_task - 3 if not is_val else num_all_task, size=(1,)).item()

        task_support_img, task_support_target = [], []
        task_query_img, task_query_target = [], []

        for _ in range(inner_batch):
            inner_support_img, inner_support_target = [], []
            inner_query_img, inner_query_target = [], []

            for class_idx in range(num_class):
                class_indices = torch.where(
                    target == task_base * num_class + class_idx)[0]

                support_indices = select(class_indices, k_support)
                query_indices = select(class_indices, k_query)

                inner_support_img.extend(img[support_indices])
                inner_support_target.extend([class_idx] * k_support)

                inner_query_img.extend(img[query_indices])
                inner_query_target.extend([class_idx] * k_query)

            task_support_img.append(torch.stack(inner_support_img))
            task_support_target.append(torch.tensor(inner_support_target))

            task_query_img.append(torch.stack(inner_query_img))
            task_query_target.append(torch.tensor(inner_query_target))

        support_imgs.append(torch.stack(task_support_img))
        support_targets.append(torch.stack(task_support_target))

        query_imgs.append(torch.stack(task_query_img))
        query_targets.append(torch.stack(task_query_target))

    return (
        torch.stack(support_imgs),
        torch.stack(support_targets),
        torch.stack(query_imgs),
        torch.stack(query_targets),
    )
