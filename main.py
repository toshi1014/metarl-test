# https://github.com/toshiouchi/MAML_ImgClassification.git

import os
import pickle
import torch
import numpy as np
from torchvision import datasets, transforms
from build_task_dataset import build_task_dataset, create_batch_of_tasks, random_seed
from maml import MAML
from train import adaptation, validation


def setup_directories():
    os.makedirs("model/", exist_ok=True)
    os.makedirs("log/", exist_ok=True)


def initialize_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    return device


def prepare_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    download_dir = "./data"
    trainset = datasets.CIFAR100(
        download_dir, train=True, transform=transform, download=True)
    evalset = datasets.CIFAR100(
        download_dir, train=False, transform=transform, download=True)

    return extract_subset(trainset, 10000), extract_subset(evalset, 5000)


def extract_subset(dataset, max_samples):
    images, targets = zip(*[(img, label)
                          for img, label in dataset][:max_samples])
    return torch.stack(images), torch.tensor(targets)


def save_model(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(path, 'model.pth'))


def save_results(results, path):
    with open(f"{path}.pkl", "wb") as file:
        pickle.dump(results, file)


def build_tasks(img, target, num_classes, num_task, k_support, k_query, is_val, outer_batch):
    return [
        build_task_dataset(
            img, target,
            num_all_task=num_classes // num_task,
            num_task=num_task, k_support=k_support,
            k_query=k_query, num_class=5,
            inner_batch=3 if not is_val else 1,
            is_val=is_val
        )
        for _ in range(outer_batch)
    ]


def main():
    setup_directories()
    device = initialize_device()

    model = MAML().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_data, val_data = prepare_datasets()
    train_img, train_target = train_data
    val_img, val_target = val_data

    num_classes = 100
    outer_batch = 5
    epochs = 300

    results = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    validation_tasks = build_tasks(
        val_img, val_target, num_classes, 5, 20, 20, True, outer_batch)

    global_step = 0
    for epoch in range(epochs):
        training_tasks = create_batch_of_tasks(build_tasks(
            train_img, train_target, num_classes, 5, 10, 15, False, outer_batch))

        for step, train_task in enumerate(training_tasks):
            loss, acc = adaptation(
                model, optimizer, train_task, loss_fn, train_step=5, train=True, device=device)
            results["train_loss"].append(loss)
            results["train_acc"].append(acc)
            print(
                f"Epoch: {epoch}, Step: {step}, Train Loss: {loss}, Train Acc: {acc}")

            if global_step % 20 == 0:
                print("------ Validation ------")
                val_results = [
                    validation(model, val_task, loss_fn,
                               train_step=10, device=device)
                    for val_task in validation_tasks
                ]
                val_loss = np.mean([res[0] for res in val_results])
                val_acc = np.mean([res[1] for res in val_results])
                results["val_loss"].append(val_loss)
                results["val_acc"].append(val_acc)
                print(
                    f"Validation Loss: {val_loss}, Validation Acc: {val_acc}")

            global_step += 1

    save_model(epoch, model, optimizer, loss, "model/")
    save_results(results, "log/results")


if __name__ == "__main__":
    main()

# 例を使った理解
# タスクの具体例
# 例えば、10クラスの画像認識問題を考えます。このとき、それぞれのタスクが異なるクラスのサブセットを扱うとします：

# タスク1: 猫、犬、鳥を分類。
# タスク2: 車、自転車、トラックを分類。
# タスク3: 花（バラ、チューリップ、ヒマワリ）を分類。
# 各タスクにおいて、以下の2種類のデータが与えられます：

# サポートセット

# 各クラスの少数サンプル（例: 1〜5枚）。
# モデルがタスクに適応するためのデータ。
# クエリセット

# モデルがサポートセットから学んだ内容をテストするための新しいデータ。
