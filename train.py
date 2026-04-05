import time
from options.train_options import TrainOptions  # 导入训练配置选项
from datasets import create_dataset  # 导入数据集创建函数
from models import create_model  # 导入模型创建函数
from utils.visualizer import Visualizer  # 导入可视化工具
import numpy as np

from models.CDP_UIE.Public.util.AverageMeter import AverageMeter  # 导入用于计算平均值的工具

if __name__ == '__main__':
    # 解析训练配置选项
    opt = TrainOptions().parse()
    print(opt)  # 打印配置选项

    # 根据配置选项创建数据集
    dataset = create_dataset(opt)
    dataset_size = len(dataset)  # 获取数据集的样本数量
    print('The number of training images = %d' % dataset_size)

    # 根据配置选项创建模型
    model = create_model(opt)
    model.setup(opt)  # 初始化模型：加载网络、打印网络结构、创建学习率调度器等
    visualizer = Visualizer(opt)  # 创建可视化工具，用于显示和保存图像及图表
    total_iters = 0  # 记录总的训练迭代次数

    best_loss = 9999.0  # 初始化最佳损失值为一个较大的数
    best_epoch = 0  # 记录最佳损失值对应的 epoch

    # 开始训练循环
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  #固定学习率阶段（默认75）+衰减阶段（默认75）
        losses_cnt = AverageMeter()  # 用于计算当前 epoch 的平均损失

        epoch_start_time = time.time()  # 记录当前 epoch 的开始时间
        epoch_iter = 0  # 记录当前 epoch 的迭代次数，每 epoch 开始时重置为 0
        visualizer.reset()  # 重置可视化工具，确保每个 epoch 至少保存一次结果到 HTML
        model.set_epoch(epoch)  # 设置当前 epoch（某些模型可能需要）

        # 遍历数据集中的每个批次
        for i, data in enumerate(dataset):
            iter_start_time = time.time()  # 记录当前迭代的开始时间

            total_iters += opt.batch_size  # 更新总迭代次数
            epoch_iter += opt.batch_size  # 更新当前 epoch 的迭代次数
            model.set_input(data)  # 将数据输入模型并进行预处理
            model.optimize_parameters()  # 计算损失、获取梯度、更新网络权重

            # 每隔 display_freq 次迭代，显示图像并保存结果
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0  # 判断是否需要保存结果到 HTML
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 每隔 print_freq 次迭代，打印训练损失并保存日志信息
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()  # 获取当前损失值
                t_comp = (time.time() - iter_start_time) / opt.batch_size  # 计算每次迭代的平均时间
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)  # 打印当前损失

            # 计算当前批次的总损失
            lss = 0.
            for k, v in model.get_current_losses().items():
                # print(f"Loss {k}: {v}")  # 打印每个损失项
                lss += v
            losses_cnt.update(lss)  # 更新平均损失

        # 如果当前 epoch 的平均损失为 NaN，跳出训练循环
        if np.isnan(losses_cnt.avg):
            visualizer.print_msg("losses_cnt.avg is nan, jump out loop of epoch")
            break

        # 如果当前 epoch 的平均损失优于最佳损失，更新最佳损失并保存模型
        if losses_cnt.avg < best_loss:
            best_loss = losses_cnt.avg
            best_epoch = epoch
            model.save_networks("best")  # 保存当前模型为最佳模型

        # 每隔 save_epoch_freq 个 epoch，保存模型
        if epoch % opt.save_epoch_freq == 0:
            visualizer.print_msg('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')  # 保存当前模型为最新模型
            model.save_networks(epoch)  # 按 epoch 编号保存模型

        # 获取当前损失并添加平均损失
        losses = model.get_current_losses()
        losses['loss_avg'] = losses_cnt.avg

        # 如果配置了显示 ID，绘制当前损失图表
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, None, losses)

        # 打印当前 epoch 的结束信息
        visualizer.print_msg('End Epoch %d. Avg Loss: %f -------- ' % (epoch, losses_cnt.avg))
        visualizer.print_msg('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # 更新学习率（在每个 epoch 结束时）
        model.update_learning_rate()

    # 训练结束，打印最佳损失和对应的 epoch
    visualizer.print_msg('Finish Training. best_loss: %f, best_epoch: %d' % (best_loss, best_epoch))

    # 打印训练完成信息
    visualizer.print_msg('Train Done!')