# Import modules we need
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

import pysnooper

from cvlib.classify import Classifier


# 这个数据集由很多不同种类的字符组成，每个字符会给20个样例
class Omniglot(Dataset):
    def __init__(self, data_dir, k_shot, q_query):
        # 这里读到的是文件夹，一个文件夹里的字符都是同一个的不同写法
        self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_shot + q_query

    def __getitem__(self, idx):
        # 根据index选择一个目录，然后从中随机抽取n个图作为样本
        sample = np.arange(20)
        np.random.shuffle(sample)  # 這裡是為了等一下要 random sample 出我們要的 character
        img_path = self.file_list[idx]
        img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
        img_list.sort()
        # TODO: 先抽取再转换
        # 1 * 28 * 28
        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        # 每個 character，随机抽取 k_shot + q_query 個
        # n * 1 * 28 * 28
        imgs = torch.stack(imgs)[sample[:self.n]]
        return imgs

    def __len__(self):
        return len(self.file_list)


class SubModel(nn.Module):

    def __init__(self, meta_model: nn.Module):
        modules = list(meta_model.named_modules())
        parameters = list(meta_model.named_parameters())
        pass


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


def MAML(model: nn.Module, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=1, inner_lr=0.4, train=True):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [batch_size * n_way, k_shot + q_query, 1, 28, 28]
    n_way: 每個分類的 task 要有幾個 class
    k_shot: 每個類別在 training 的時候會有多少張照片
    q_query: 在 testing 時，每個類別會用多少張照片 update
    """
    batch_x = x.reshape(-1, n_way, k_shot + q_query, 1, 28, 28)

    criterion = loss_fn
    task_loss = []  # 這裡面之後會放入每個 task 的 loss
    task_acc = []  # 這裡面之後會放入每個 task 的 loss

    for meta_batch in batch_x:
        train_set = meta_batch[:, :k_shot].squeeze(axis=1)  # train_set 是我們拿來 update inner loop 參數的 data
        val_set = meta_batch[:, k_shot:].squeeze(axis=1)  # val_set 是我們拿來 update outer loop 參數的 data

        sub_model = SubModel(model)

        for inner_step in range(inner_train_step):  # 這個 for loop 是 Algorithm2 的 line 7~8
            # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
            # 所以我們還是用 for loop 來寫
            train_label = create_label(n_way, k_shot)
            logits = sub_model(train_set)
            loss = criterion(logits, train_label)
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True)  # 這裡是要計算出 loss 對 θ 的微分 (∇loss)
            fast_weights = OrderedDict((name, param - inner_lr * grad)
                                       for ((name, param), grad) in
                                       zip(fast_weights.items(), grads))  # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'

        val_label = create_label(n_way, q_query)
        logits = model.functional_forward(val_set, fast_weights)  # 這裡用 val_set 和 θ' 算 logit
        loss = criterion(logits, val_label)  # 這裡用 val_set 和 θ' 算 loss
        task_loss.append(loss)  # 把這個 task 的 loss 丟進 task_loss 裡面
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()  # 算 accuracy
        task_acc.append(acc)

    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()  # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc


def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    # 把单条样本组装成mini_batch
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way, k_shot+q_query, 1, 28, 28]
        except StopIteration:
            iterator = iter(data_loader)
            task_data = iterator.next()
        # 这里的一组数据是n_ways条，同一个类别拆成support set和query set
        train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
        val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
        task_data = torch.cat((train_data, val_data), 0)
        data.append(task_data)
    return torch.stack(data), iterator


def test_dataset():
    train_data_path = './data/Omniglot/images_background/'
    data_set = Omniglot(train_data_path, 1, 1)
    for i in data_set:
        print(i)


def meta_demo(train_data_path, test_data_path):
    n_way = 5
    k_shot = 1
    q_query = 1
    inner_train_step = 1
    inner_lr = 0.4
    meta_lr = 0.001
    meta_batch_size = 32
    max_epoch = 20
    eval_batches = test_batches = 20

    # dataset = Omniglot(train_data_path, k_shot, q_query)
    train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200, 656])

    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=meta_batch_size * n_way,
                              num_workers=8,
                              shuffle=True,
                              drop_last=True,
                              )
    val_loader = DataLoader(val_set,
                            batch_size=n_way,
                            num_workers=8,
                            shuffle=True,
                            drop_last=True,
                            )
    test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query),
                             batch_size=n_way,
                             num_workers=8,
                             shuffle=True,
                             drop_last=True,
                             )

    meta_model = Classifier(1, n_way)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(max_epoch):
        print("Epoch %d" % (epoch))
        train_meta_loss = []
        train_acc = []
        for data in tqdm(train_loader):  # 這裡的 step 是一次 meta-gradinet update step
            meta_loss, acc = MAML(meta_model, optimizer, data, n_way, k_shot, q_query, loss_fn)
            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)
        print("  Loss    : ", np.mean(train_meta_loss))
        print("  Accuracy: ", np.mean(train_acc))

        # 每個 epoch 結束後，看看 validation accuracy 如何
        # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的
        val_acc = []
        for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
            x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
            _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=3,
                          train=False)  # testing時，我們更新三次 inner-step
            val_acc.append(acc)
        print("  Validation accuracy: ", np.mean(val_acc))


if __name__ == '__main__':
    meta_demo(
        train_data_path='./data/Omniglot/images_background/',
        test_data_path='./data/Omniglot/images_evaluation/',
    )
