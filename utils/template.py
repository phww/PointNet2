import time
import torch
from torchsummary import summary
import os
import os.path as osp
import sys


class Logger(object):
    """将stdout重定位到log文件和控制台
        即print时会将信息打印在控制台的
        同时也将信息保存在log文件中
    """

    def __init__(self, filename):
        self.console = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.console.write(message)
        self.log.write(message)

    # 清空log文件中的内容
    def clean(self):
        self.log.truncate(0)

    def flush(self):
        pass


class TemplateModel:
    def __init__(self):
        # 必须设定
        # 模型架构
        # 将模型和优化器以list保存，方便对整个模型的多个部分设定对应的优化器
        self.model_list = None  # 模型的list
        self.optimizer_list = None  # 优化器的list
        self.criterion = None
        # 数据集
        self.train_loader = None
        self.test_loader = None

        # 下面的可以不设定
        # tensorboard
        self.writer = None  # 推荐设定
        # 训练时print的间隔
        self.log_per_step = 5  # 推荐按数据集大小设定
        # 推荐设置学习率衰减
        self.lr_scheduler_type = None  # None "metric" "loss"
        # check_point 目录
        self.ckpt_dir = "./check_point/" + time.strftime("%Y-%m-%d::%H:%M:%S")

        # 不用设定
        # 训练状态
        self.global_step = 0
        self.global_step_eval = 0
        self.epoch = 1
        self.best_metric = {}
        self.key_metric = None
        self.lr_scheduler_list = None
        # 运行设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def check_init(self, log_name="log.txt", clean_log=False, arg=None):
        # 检测摸板的初始状态，可以在这加上很多在训练之前的操作
        assert isinstance(self.model_list, list)
        assert isinstance(self.optimizer_list, list)
        assert self.criterion
        assert self.train_loader
        assert self.test_loader
        assert self.device
        assert self.ckpt_dir
        assert self.log_per_step
        assert self.lr_scheduler_type in [None, "metric", "loss"]
        if not osp.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # 设置lr_scheduler
        # 如果以测试集的metric为学习率改变的依据，选择mode="max";以loss为依据，选择mode="min"
        mode = "max"
        if self.lr_scheduler_type is not None:
            if self.lr_scheduler_type == "metric":
                mode = "max"
            elif self.lr_scheduler_type == "loss":
                mode = "min"
            self.lr_scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                 mode=mode,
                                                                                 factor=0.1,
                                                                                 patience=5,
                                                                                 cooldown=5,
                                                                                 min_lr=1e-9,
                                                                                 verbose=True)
                                      for optimizer in self.optimizer_list]

        # 设置log,log保存目录为"self.ckpt_dir/log_name"
        logger = Logger(os.path.join(self.ckpt_dir, log_name))
        if clean_log:
            logger.clean()
        sys.stdout = logger
        print(time.strftime("%Y-%m-%d::%H-%M-%S"))
        self.print_all_member()
        # 如果有，打印arg
        if arg is not None:
            print(15 * "=", "args", 15 * "=")
            arg_dict = arg.__dict__
            for key in arg_dict.keys():
                print(f"{key}:{arg_dict[key]}")

        # 清空cuda中的cache
        torch.cuda.empty_cache()

        for model in self.model_list:
            model.to(self.device)

    def load_state(self, fname, optim=True, lr_list=None):
        # 读取保存的模型到模板之中。如果要继续训练的模型optim=True；使用最佳模型做推断optim=False
        state = torch.load(fname)
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):  # 多卡训练
                model.module.load_state_dict(state[f'model{idx}'])
            else:  # 非多卡训练
                model.load_state_dict(state[f'model{idx}'])
            # 恢复一些状态参数
            if optim and f'optimizer_list{idx}' in state:
                self.optimizer_list[idx].load_state_dict(state[f'optimizer_list{idx}'])
                # 改变先前模型的优化器中保存的学习率
                if lr_list is not None:
                    self.change_lr(lr_list)

        self.global_step = state['global_step']
        self.global_step_eval = state["global_step_eval"]
        self.epoch = state['epoch']
        self.best_metric = state['best_metric']
        self.key_metric = state['key_metric']
        print('load model from {}'.format(fname))

    def save_state(self, fname, optim=True):
        # 保存模型，其中最佳模型不用保存优化器中的参数。
        # 而训练过程中保存的其他模型需要保存优化器中的梯度以便继续训练
        state = {}
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):
                state[f'model{idx}'] = model.module.state_dict()
            else:
                state[f'model{idx}'] = model.state_dict()
            # 训练过程中的模型除了保存模型的参数外，还要保存当前训练的状态：optim中的参数
            if optim:
                state[f'optimizer_list{idx}'] = self.optimizer_list[idx].state_dict()
        state['global_step'] = self.global_step
        state['global_step_eval'] = self.global_step_eval
        state['epoch'] = self.epoch
        state['best_metric'] = self.best_metric
        state['key_metric'] = self.key_metric
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def train_loop(self, write_params=False):
        """训练一个epoch，一般来说不用改"""
        print("*" * 15, f"epoch:{self.epoch}", "*" * 15)
        for model in self.model_list:
            model.train()

        running_loss = 0.0
        all_avg_loss = 0.0
        cnt_loss = 0
        for step, batch in enumerate(self.train_loader):
            self.global_step += 1
            batch_loss = self.loss_per_batch(batch)

            # 多个优化器需要按逆序更新每一个优化器
            for optimizer in reversed(self.optimizer_list):
                optimizer.zero_grad()

            batch_loss.backward()

            for optimizer in reversed(self.optimizer_list):
                optimizer.step()
            running_loss += batch_loss.item()

            # 记录损失除了训练刚开始时是用此时的loss外，其他都是用一批loss的平均loss
            if self.global_step == 1:
                # 为了tensorboard的曲线好看，不记录这个loss
                # if self.writer is not None:
                #     self.writer.add_scalar('train_loss', batch_loss.item(), self.global_step)
                print(f"loss:{batch_loss.item() : .5f}\t"
                      f"cur:[{step * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]")

            # 打印self.log_per_step批数据的的平均loss
            elif (step + 1) % self.log_per_step == 0:
                avg_loss = running_loss / (self.log_per_step * len(batch))
                print(f"loss:{avg_loss : .5f}\t"
                      f"cur:[{(step + 1) * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]")
                all_avg_loss += avg_loss
                cnt_loss += 1
                running_loss = 0.0

                # Tensorboard记录
                if self.writer is not None:
                    # 平均loss
                    self.writer.add_scalar('train_loss', avg_loss, self.global_step)

                    # write_params=True在Tensorboard中记录模型的参数和梯度的分布情况，但是也费时间。默认关闭
                    if write_params:
                        for model in self.model_list:
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy())
                                if value.grad is not None:  # 在FineTurn时有些参数被冻结了，没有梯度。也就不用记录了
                                    self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy())

        # 一个epoch，训练集的全部样本的平均loss
        avg_batch_loss = all_avg_loss / cnt_loss
        print(f"epoch:{self.epoch}\tavg_epoch_loss:{avg_batch_loss:.5f}")
        if self.writer is not None:
            self.writer.add_scalars("avg_epoch_loss", {"train": avg_batch_loss}, self.epoch)

    def loss_per_batch(self, batch):
        """
        计算数据集的一个batch的loss，这个部分是可能要按需求修改的部分
        Pytorch 中的loss函数中一般规定x和y都是float，而有些loss函数规定y要为long（比如经常用到的CrossEntropyLoss）
        如果官网：https://pytorch.org/docs/stable/nn.html#loss-functions 对y的数据类型有要求请做相应的修改
        这里除了CrossEntropyLoss将y的数据类型设为long外， 其他都默认x和y的数据类型为float
        """
        x, y = batch
        x = x.to(self.device, dtype=torch.float)

        # 标签y的数据类型
        y_dtype = torch.float
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            y_dtype = torch.long

        # 保证标签y至少是个列向量，即shape "B, 1"
        if y.dim == 1:
            y = y.unsqueeze(dim=1).to(self.device, dtype=y_dtype)
        else:
            y = y.to(self.device, dtype=y_dtype)

        # 若模型的输入不是一个tensor，按需求改
        pred = x
        for model in self.model_list:
            pred = model(pred)
        loss = self.criterion(pred, y)
        return loss

    def eval_loop(self, save_per_epochs=1):
        """一个epoch的评估。一般不用改"""
        print("-" * 15, "Evaluation", "-" * 15)
        # 在整个测试集上做评估，使用分批次的metric的平均值表示训练集整体的metric
        with torch.no_grad():
            for model in self.model_list:
                model.eval()

            # 验证时的loss
            running_loss = 0.0
            all_avg_loss = 0.0
            # 保存各种metric的得分
            scores = {}
            cnt = 0
            cnt_loss = 0
            for step, batch in enumerate(self.test_loader):
                cnt += 1
                self.global_step_eval += 1

                # 计算评估时多个批次的平均loss
                batch_loss = self.loss_per_batch(batch)
                running_loss += batch_loss.item()
                # 每self.log_per_step个step打印信息
                if (step + 1) % self.log_per_step == 0:
                    avg_loss = running_loss / (self.log_per_step * len(batch))
                    if self.writer is not None:
                        self.writer.add_scalar('eval_loss', avg_loss, self.global_step_eval)
                    print(f"loss:{avg_loss : .5f}\t"
                          f"cur:[{(step + 1) * self.test_loader.batch_size}]\[{len(self.test_loader.dataset)}]")
                    all_avg_loss += avg_loss
                    cnt_loss += 1
                    running_loss = 0.0

                # 分批计算metric
                if step == 0:
                    scores = self.eval_scores_per_batch(batch)  # 每个epoch初始化scores
                # 累加所有批次的metric。这里有个问题：
                # 准确率可以使用分批的准确率之和除以分批数量得到，并且与用全部数据集计算准确率是等价的
                # 但是有的metric使用一部分批次的计算出来的结果可能与使用全部数据集计算出来的结果不同
                else:
                    batch_scores = self.eval_scores_per_batch(batch)
                    for key in scores.keys():
                        scores[key] += batch_scores[key]
                if self.global_step_eval == 1:
                    self.best_metric = scores  # 第一次eval时，初始化self.best_metric

            # 整个测试集上的平均running_loss
            avg_batch_loss = all_avg_loss / cnt_loss
            # 整个测试集上的平均metric
            for key in scores.keys():
                scores[key] /= cnt

            # 根据scores[self.key_metric]来判定是否保存最佳模型.
            # self.key_metric需要在metric函数中初始化，分类任务常用self.key_metric = "acc"
            for key in scores.keys():
                # 更新所有metric的最佳结果到self.best_metric字典中
                if scores[key] >= self.best_metric[key]:
                    self.best_metric[key] = scores[key]

                    # 保存最佳模型
                    if key == self.key_metric:
                        self.save_state(osp.join(self.ckpt_dir, f'best.pth'), False)

            # 每save_per_epochs次评估就保存当前模型，这种模型一般用于继续训练
            if self.epoch % save_per_epochs == 0:
                self.save_state(osp.join(self.ckpt_dir, f'epoch{self.epoch}.pth'))

            # 打印信息:一个epoch上的平均loss和关键指标
            print(f"epoch:{self.epoch}\tavg_epoch_loss:{avg_batch_loss:.5f}")
            print(f'epoch:{self.epoch}\t{self.key_metric}:{scores[self.key_metric]:.5f}')

            # Tensorboard
            if self.writer is not None:
                # 记录每个epoch的metric
                for key in scores.keys():
                    self.writer.add_scalar(f"eval_{key}", scores[key], self.epoch)

                # 记录一个epoch中验证集中全部样本的平均loss
                self.writer.add_scalars("avg_epoch_loss", {"eval": avg_batch_loss}, self.epoch)

                # 记录lr_scheduler中学习率的变化情况
                if self.lr_scheduler_list is not None:
                    for i, lr_scheduler in enumerate(self.lr_scheduler_list):
                        self.writer.add_scalar(f"lr_scheduler{i}",
                                               self.optimizer_list[i].param_groups[0]["lr"],
                                               self.epoch)
                        if self.lr_scheduler_type == "metric":
                            lr_scheduler.step(scores[self.key_metric])
                        elif self.lr_scheduler_type == "loss":
                            lr_scheduler.step(avg_batch_loss)
            self.epoch += 1
        return scores[self.key_metric]

    # 以下eval_scores_per_batch()和metric()，有时要按需求修改
    def eval_scores_per_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        pred = x
        for model in self.model_list:
            pred = model(pred)
        scores_pre_batch = self.metric(pred, y)
        return scores_pre_batch

    def metric(self, pred, y):
        """
        不同任务的性能指标太难统一了，这里只是实现了多分类任务求准确率的方法。其他任务请按需求继承
        这个类的时候再重载这个metric函数，注意返回数据类型为字典,且一定要有self.key_metric这个
        指标，因为self.key_metric用于保存训练过程中的最优模型.这个模板使用分批计算metric再求全
        部批次的平均值的策略得到整体的metric。不会将全部的预测和ground truth保存在preds和ys中
        然后在cpu上进行预测。因为如果测试集或验证集太大（>50000）可能CPU内存装不下，训练会报错.但
        是有的metric可能不能使用分批得到的metric求平均来表示整体的metric,按需求改吧
        Args:
            pred: torch.tensor
                测试集或验证集的一个批次的预测
            y: torch.tensor
                测试集或验证集的一个批次的ground truth

        Returns:
            scores：dict
                各种性能指标的字典，务必要有scores[self.key_metric]

        """
        # 初始化self.key_metric
        self.key_metric = "acc"

        scores = {}
        correct = (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
        scores[self.key_metric] = correct / self.test_loader.batch_size
        return scores

    # 有时要按需求修改
    def inference(self, x):
        x = x.to(self.device)
        for model in self.model_list:
            x = model(x)
        return x

    def get_model_info(self, fake_inp):
        # 输出模型信息和在Tensorboard中绘制模型计算图
        print(15 * "=", "model info", 15 * "=")
        if self.writer is not None:
            for model in self.model_list:
                # self.writer.add_graph(model, fake_inp.to(self.device))
                # summary对transformer有BUG
                print(summary(model, batch_size=32, input_size=fake_inp.shape[1:], device=self.device))

    def num_parameters(self):
        num = 0
        for model in self.model_list:
            num += sum([p.data.nelement() for p in model.parameters()])
        return num

    def print_best_metrics(self):
        for key in self.best_metric.keys():
            print(f"{key}:\t{self.best_metric[key]}")

    def print_final_lr(self):
        for i, optimizer in enumerate(self.optimizer_list):
            print(f"final_lr_{i}:{optimizer.param_groups[0]['lr']}")

    def print_all_member(self, print_model=False):
        print(15 * "=", "template config", 15 * "=")
        # 不重要，不需要打印的信息
        except_member = ["best_metric", "key_metric", "train_loader", "test_loader", "writer"]
        # 模型信息太长了，选择打印
        if not print_model:
            except_member.append("model_list")
        for name, value in vars(self).items():
            if name not in except_member:
                print(f"{name}:{value}")

    def change_lr(self, lr_list):
        """
        改变优化器中记录的学习率，主要用于使用已有的模型继续训练时。
        指定新的学习率，而不是已有模型的优化器中保存的学习率。
        """
        for i, optimizer in enumerate(self.optimizer_list):
            optimizer.param_groups[0]['lr'] = lr_list[i]
