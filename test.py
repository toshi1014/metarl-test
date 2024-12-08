import torch
import numpy as np
import utils
import prepare
import maml
import train
import torchvision


def test():
    utils.setup_directories(["log"])
    device = utils.initialize_device()

    model = maml.MAML().to(device)
    model = utils.load_model(model, "model/model.pth", device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_img, test_target = utils.prepare_dataset(
        torchvision.datasets.CIFAR100,
        "./data",
        train=False,
        transform=transform,
        max_samples=10000,
    )

    test_tasks = prepare.build_tasks(
        img=test_img,
        target=test_target,
        num_classes=20,
        num_class=5,
        num_task=20,
        k_support=20,
        k_query=20,
        is_val=True,
        outer_batch=10,
    )
    db_test = prepare.create_batch_of_tasks(
        test_tasks, is_shuffle=False, outer_batch_size=10)

    acc_all_test, loss_all_test = [], []

    for loop, test_task in enumerate(db_test):
        loss, acc = train.validation(
            model, test_task, loss_fn, train_step=10, device=device)
        acc_all_test.append(acc)
        loss_all_test.append(loss)
        print(f"Loop: {loop}, Test Loss: {np.mean(loss_all_test):.4f}, Acc: {np.mean(acc_all_test):.4f}") \

    utils.save_results(
        {'test_loss': loss_all_test, 'test_acc': acc_all_test},
        "log/test_results"
    )


if __name__ == "__main__":
    test()
