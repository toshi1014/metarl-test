import torch
import numpy as np
import torchvision
import utils
import prepare
import maml
import train


def main():
    utils.setup_directories(["model", "log"])
    device = utils.initialize_device()

    model = maml.MAML().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_img, train_target = utils.prepare_dataset(
        torchvision.datasets.CIFAR100, "./data", train=True, transform=transform, max_samples=10000)
    val_img, val_target = utils.prepare_dataset(
        torchvision.datasets.CIFAR100, "./data", train=False, transform=transform, max_samples=5000)

    NUM_CLASSES = 100
    NUM_CLASS = 3
    NUM_TASK = 5
    OUTER_BATCH = 5
    EPOCHS = 300

    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    validation_tasks = prepare.build_tasks(
        img=val_img,
        target=val_target,
        num_classes=NUM_CLASSES,
        num_class=NUM_CLASS,
        num_task=NUM_TASK,
        k_support=20,
        k_query=20,
        is_val=True,
        outer_batch=OUTER_BATCH,
    )

    global_step = 0
    for epoch in range(EPOCHS):
        training_tasks = prepare.create_batch_of_tasks(
            prepare.build_tasks(
                img=train_img,
                target=train_target,
                num_classes=NUM_CLASSES,
                num_class=NUM_CLASS,
                num_task=NUM_TASK,
                k_support=10,
                k_query=15,
                is_val=False,
                outer_batch=OUTER_BATCH,
            )
        )

        for step, train_task in enumerate(training_tasks):
            loss, acc = train.adaptation(
                model,
                optimizer,
                train_task,
                loss_fn,
                train_step=5,
                device=device,
            )
            results["train_loss"].append(loss)
            results["train_acc"].append(acc)
            print(f"Epoch: {epoch}, Step: {step}, Train Loss: {loss}, Train Acc: {acc}") \

            if global_step % 20 == 0:
                val_results = [
                    train.validation(
                        model,
                        val_task,
                        loss_fn,
                        train_step=10,
                        device=device,
                    )
                    for val_task in validation_tasks
                ]
                val_loss = np.mean([res[0] for res in val_results])
                val_acc = np.mean([res[1] for res in val_results])
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)
                print(f"\n\nValidation Loss: {val_loss}, Validation Acc: {val_acc}\n\n") \

            global_step += 1

    utils.save_model(epoch, model, optimizer, loss, "model/")
    utils.save_results(results, "log/results")


if __name__ == "__main__":
    main()
