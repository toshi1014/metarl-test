import torch
from collections import OrderedDict


# Constants
N_CLASS = 3  # Number of color channels (RGB)
IMG_SIZE = 32  # Image width and height


def adaptation_per_task(model, dataset, loss_fn, train_step, device, create_graph=False):
    x_train, y_train, x_val, y_val = dataset

    lr_inner = 0.001
    task_accuracies = []
    total_loss = 0

    # Initialize task-specific weights
    weights = OrderedDict(model.named_parameters())

    # Support data
    support_x = x_train.to(device)
    support_y = y_train.to(device)

    # Inner-loop adaptation
    for _ in range(train_step):
        logits = model.adaptation(
            support_x.view(-1, N_CLASS, IMG_SIZE, IMG_SIZE), weights)
        task_loss = loss_fn(logits, support_y.view(-1))
        gradients = torch.autograd.grad(
            task_loss,
            weights.values(),
            create_graph=create_graph,
        )
        weights = OrderedDict(
            (name, param - lr_inner * grad)
            for ((name, param), grad) in zip(weights.items(), gradients)
        )

    # Query data evaluation
    query_x = x_val.to(device)
    query_y = y_val.to(device)
    query_x = query_x.view(-1, N_CLASS, IMG_SIZE, IMG_SIZE)
    query_y = query_y.view(-1)

    logits = model.adaptation(query_x, weights)
    query_loss = loss_fn(logits, query_y)

    # Accumulate loss and accuracy
    total_loss += query_loss.item()
    predicted_labels = torch.argmax(logits, dim=1)
    task_acc = torch.eq(predicted_labels, query_y).float().mean().item()
    task_accuracies.append(task_acc)

    return task_accuracies, query_loss,  total_loss


def adaptation(model, optimizer, batch, loss_fn, train_step, device):
    num_task = batch[0].size(0)

    # Outer gradient accumulation (used during training)
    aggregated_gradients = None
    outer_loss = 0

    for task_idx in range(num_task):  # Loop over tasks
        task_accuracies, query_loss, total_loss = adaptation_per_task(
            model,
            [batch[i][task_idx] for i in range(4)],
            loss_fn,
            train_step,
            device,
            create_graph=True,
        )

        # Outer-loop update during training
        task_gradients = torch.autograd.grad(
            query_loss,
            model.parameters(),
            retain_graph=False,
        )

        if aggregated_gradients is None:
            aggregated_gradients = list(task_gradients)
        else:
            aggregated_gradients = [
                g1 + g2 for g1, g2 in zip(aggregated_gradients, task_gradients)
            ]
            outer_loss += query_loss

    # Outer-loop update
    for param, grad in zip(model.parameters(), aggregated_gradients):
        param.grad = grad
    optimizer.step()
    optimizer.zero_grad()

    avg_outer_loss = outer_loss.item() / num_task
    avg_accuracy = torch.tensor(task_accuracies).mean().item()
    return avg_outer_loss, avg_accuracy


def validation(model, batch, loss_fn, train_step, device):
    num_task = batch[0].size(0)

    for task_idx in range(num_task):  # Loop over tasks
        task_accuracies, query_loss, total_loss = adaptation_per_task(
            model,
            [batch[i][task_idx] for i in range(4)],
            loss_fn,
            train_step,
            device,
        )

    avg_outer_loss = total_loss / num_task
    avg_accuracy = torch.tensor(task_accuracies).mean().item()
    return avg_outer_loss, avg_accuracy
