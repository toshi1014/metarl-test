import torch
from collections import OrderedDict

# Constants
N_CLASS = 3  # Number of color channels (RGB)
IMG_SIZE = 32  # Image width and height


def adaptation(model, optimizer, batch, loss_fn, train_step, train, device):
    """
    Adaptation process for meta-learning with support and query datasets.

    Args:
        model: Meta-learning model.
        optimizer: Optimizer for outer updates.
        batch: Tuple containing support and query data (images and labels).
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        train_step: Number of inner-loop training steps.
        train: Boolean indicating whether to perform training or not.
        device: Computation device (CPU/GPU).

    Returns:
        Average loss and accuracy across tasks.
    """
    x_train, y_train, x_val, y_val = batch
    task_accuracies = []
    lr_inner = 0.001
    outer_loss = 0

    weights = OrderedDict(model.named_parameters())

    for task_idx in range(x_train.size(0)):  # Loop over tasks
        # Support data
        support_x, support_y = x_train[task_idx].to(
            device), y_train[task_idx].to(device)
        task_loss = 0

        # Inner-loop adaptation
        for step in range(train_step):
            for x, y in zip(support_x, support_y):
                logits = model(x)
                task_loss += loss_fn(logits, y)

        task_loss /= (len(support_x) * train_step)

        # Compute gradients and update task-specific weights
        gradients = torch.autograd.grad(task_loss, weights.values())
        adapted_weights = OrderedDict(
            (name, param - lr_inner * grad) for ((name, param), grad) in zip(weights.items(), gradients)
        )

        # Query data
        query_x, query_y = x_val[task_idx].to(
            device), y_val[task_idx].to(device)
        query_x = query_x.view(-1, N_CLASS, IMG_SIZE, IMG_SIZE)
        query_y = query_y.view(-1)

        logits = model.adaptation(query_x, adapted_weights)
        query_loss = loss_fn(logits, query_y)

        if train:
            task_gradients = torch.autograd.grad(query_loss, weights.values())
            if task_idx == 0:
                aggregated_gradients = list(task_gradients)
            else:
                aggregated_gradients = [
                    g1 + g2 for g1, g2 in zip(aggregated_gradients, task_gradients)
                ]
            outer_loss += query_loss

        predicted_labels = torch.argmax(logits, dim=1)
        task_acc = torch.eq(predicted_labels, query_y).float().mean().item()
        task_accuracies.append(task_acc)

    # Outer-loop update
    if train:
        for param, grad in zip(model.parameters(), aggregated_gradients):
            param.grad = grad
        optimizer.step()
        optimizer.zero_grad()

    avg_outer_loss = outer_loss.item() / x_train.size(0)
    avg_task_accuracy = torch.tensor(task_accuracies).mean().item()

    return avg_outer_loss, avg_task_accuracy


def validation(model, batch, loss_fn, train_step, device):
    """
    Validation routine to evaluate model performance on unseen tasks.

    Args:
        model: Meta-learning model.
        batch: Tuple containing support and query data (images and labels).
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        train_step: Number of inner-loop training steps.
        device: Computation device (CPU/GPU).

    Returns:
        Average loss and accuracy across validation tasks.
    """
    x_train, y_train, x_val, y_val = batch
    lr_inner = 0.001
    task_accuracies = []
    total_loss = 0

    for task_idx in range(x_train.size(0)):  # Loop over tasks
        weights = OrderedDict(model.named_parameters())
        support_x, support_y = x_train[task_idx].to(
            device), y_train[task_idx].to(device)

        # Inner-loop adaptation
        for step in range(train_step):
            logits = model.adaptation(
                support_x.view(-1, N_CLASS, IMG_SIZE, IMG_SIZE), weights)
            loss = loss_fn(logits, support_y.view(-1))
            gradients = torch.autograd.grad(loss, weights.values())
            weights = OrderedDict(
                (name, param - lr_inner * grad) for ((name, param), grad) in zip(weights.items(), gradients)
            )

        # Query data evaluation
        query_x, query_y = x_val[task_idx].to(
            device), y_val[task_idx].to(device)
        query_x = query_x.view(-1, N_CLASS, IMG_SIZE, IMG_SIZE)
        query_y = query_y.view(-1)

        with torch.no_grad():
            logits = model.adaptation(query_x, weights)
            task_loss = loss_fn(logits, query_y)
            total_loss += task_loss.item()

            predicted_labels = torch.argmax(logits, dim=1)
            task_acc = torch.eq(
                predicted_labels, query_y).float().mean().item()
            task_accuracies.append(task_acc)

    avg_loss = total_loss / x_train.size(0)
    avg_accuracy = torch.tensor(task_accuracies).mean().item()

    return avg_loss, avg_accuracy


def test_model(model, batch, loss_fn, train_step, device):
    """
    Test routine for evaluating the model on test tasks.

    Args:
        model: Meta-learning model.
        batch: Tuple containing support and query data (images and labels).
        loss_fn: Loss function (e.g., CrossEntropyLoss).
        train_step: Number of inner-loop training steps.
        device: Computation device (CPU/GPU).

    Returns:
        Average loss and accuracy across test tasks.
    """
    return validation(model, batch, loss_fn, train_step, device)
