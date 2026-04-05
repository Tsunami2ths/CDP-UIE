# import os
# import torch
# from collections import OrderedDict
# from abc import ABC, abstractmethod
# from torch.optim import lr_scheduler
#
# class BaseModel(ABC):
#     """This class is an abstract base class (ABC) for models.
#     To create a subclass, you need to implement the following five functions:
#         -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
#         -- <set_input>:                     unpack data from dataset and apply preprocessing.
#         -- <forward>:                       produce intermediate results.
#         -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
#         -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
#     """
#
#     def __init__(self, opt):
#         """Initialize the BaseModel class.
#
#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#
#         When creating your custom class, you need to implement your own initialization.
#         In this fucntion, you should first call <BaseModel.__init__(self, opt)>
#         Then, you need to define four lists:
#             -- self.loss_names (str list):          specify the training losses that you want to plot and save.
#             -- self.model_names (str list):         specify the images that you want to display and save.
#             -- self.visual_names (str list):        define networks used in our training.
#             -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
#         """
#         self.opt = opt
#         self.gpu_ids = opt.gpu_ids
#         self.isTrain = opt.isTrain
#         self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
#         self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
#         if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
#             torch.backends.cudnn.benchmark = True
#         self.loss_names = []
#         self.model_names = []
#         self.visual_names = []
#         self.optimizers = []
#         self.image_paths = []
#         self.metric = 0  # used for learning rate policy 'plateau'
#         # self.current_epoch = opt.epoch_count
#
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         """Add new model-specific options, and rewrite default values for existing options.
#
#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
#
#         Returns:
#             the modified parser.
#         """
#         return parser
#
#     @abstractmethod
#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.
#
#         Parameters:
#             input (dict): includes the data itself and its metadata information.
#         """
#         pass
#
#     @abstractmethod
#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         pass
#
#     @abstractmethod
#     def optimize_parameters(self):
#         """Calculate losses, gradients, and update network weights; called in every training iteration"""
#         pass
#
#
#     def setup(self, opt):
#         """
#         加载网络、创建学习率调度器并打印网络结构信息。
#
#         参数:
#             opt (Option class): 包含所有实验配置的选项类，需继承自 BaseOptions。
#         """
#         # 如果是训练模式，创建学习率调度器
#         if self.isTrain:
#             # 为每个优化器创建对应的学习率调度器
#             self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
#
#         # 如果当前模式不是训练模式，或者指定了 --continue_train 参数，则加载检查点文件
#         if not self.isTrain or opt.continue_train:
#             # 根据 opt.load_iter 和 opt.epoch 决定加载的模型文件后缀
#             load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
#             # 打印加载模型权重的日志
#             print(f"[INFO] 尝试加载模型权重文件，后缀为: {load_suffix}")
#             self.load_networks(load_suffix)
#             print(f"[INFO] 成功加载模型权重文件，后缀为: {load_suffix}")
#
#
#         # 打印网络结构信息，详细程度由 opt.verbose 决定
#         self.print_networks(opt.verbose)
#
#     def eval(self):
#         """Make models eval mode during test time"""
#         for name in self.model_names:
#             if isinstance(name, str):
#                 net = getattr(self, 'net' + name)
#                 net.eval()
#
#     def test(self):
#         """Forward function used in test time.
#
#         This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
#         It also calls <compute_visuals> to produce additional visualization results
#         """
#         with torch.no_grad():
#             self.forward()
#             self.compute_visuals()
#
#     def compute_visuals(self):
#         """Calculate additional output images for visdom and HTML visualization"""
#         pass
#
#     def get_image_paths(self):
#         """ Return image paths that are used to load current data"""
#         return self.image_paths
#
#     def update_learning_rate(self):
#         """Update learning rates for all the networks; called at the end of every epoch"""
#         for scheduler in self.schedulers:
#             if self.opt.lr_policy == 'plateau':
#                 scheduler.step(self.metric)
#             else:
#                 scheduler.step()
#
#         lr = self.optimizers[0].param_groups[0]['lr']
#         print('learning rate = %.7f' % lr)
#
#     def get_current_visual_names(self):
#         """Return visual_names. test.py will use it to create multi-pred output dir"""
#         return self.visual_names
#
#     def get_current_visuals(self):
#         """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
#         visual_ret = OrderedDict()
#         for name in self.visual_names:
#             if isinstance(name, str):
#                 visual_ret[name] = getattr(self, name)
#         return visual_ret, self.image_paths
#
#     def get_current_losses(self):
#         """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
#         errors_ret = OrderedDict()
#         for name in self.loss_names:
#             if isinstance(name, str):
#                 errors_ret[name] = float(getattr(self, 'loss_' + name, 0.0))  # float(...) works for both scalar tensor and float number
#         return errors_ret
#
#     def get_current_losses_tensor(self):
#         """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
#         errors_ret = OrderedDict()
#         for name in self.loss_names:
#             if isinstance(name, str):
#                 errors_ret[name] = getattr(self, 'loss_' + name, 0.0)  # float(...) works for both scalar tensor and float number
#         return errors_ret
#
#     def save_networks(self, epoch):
#         """Save all the networks to the disk.
#
#         Parameters:
#             epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
#         """
#         for name in self.model_names:
#             if isinstance(name, str):
#                 save_filename = '%s_net_%s.pth' % (epoch, name)
#                 save_path = os.path.join(self.save_dir, save_filename)
#                 net = getattr(self, 'net' + name)
#
#                 if len(self.gpu_ids) > 0 and torch.cuda.is_available():
#                     try:
#                         torch.save(net.module.cpu().state_dict(), save_path)
#                     except AttributeError as e:
#                         torch.save(net.cpu().state_dict(), save_path)
#                     net.cuda(self.gpu_ids[0])
#                 else:
#                     torch.save(net.cpu().state_dict(), save_path)
#
#     def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
#         """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
#         key = keys[i]
#         if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
#             if module.__class__.__name__.startswith('InstanceNorm') and \
#                     (key == 'running_mean' or key == 'running_var'):
#                 if getattr(module, key) is None:
#                     state_dict.pop('.'.join(keys))
#             if module.__class__.__name__.startswith('InstanceNorm') and \
#                (key == 'num_batches_tracked'):
#                 state_dict.pop('.'.join(keys))
#         else:
#             self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
#
#     def load_networks1(self, epoch, model_names):
#         """Load all the networks from the disk.
#
#         Parameters:
#             epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
#         """
#         for name in model_names:
#             if isinstance(name, str):
#                 load_filename = '%s_net_%s.pth' % (epoch, name)
#                 load_path = os.path.join(self.save_dir, load_filename)
#                 net = getattr(self, 'net' + name)
#                 if isinstance(net, torch.nn.DataParallel):
#                     net = net.module
#                 print('loading the model from %s' % load_path)
#                 # if you are using PyTorch newer than 0.4 (e.g., built from
#                 # GitHub source), you can remove str() on self.device
#                 state_dict = torch.load(load_path, map_location=str(self.device))
#                 if hasattr(state_dict, '_metadata'):
#                     del state_dict._metadata
#
#                 # patch InstanceNorm checkpoints prior to 0.4
#                 for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
#                     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
#                 net.load_state_dict(state_dict)
#
#     def load_networks(self, epoch):
#         """Load all the networks from the disk.
#
#         Parameters:
#             epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
#         """
#         for name in self.model_names:
#             if isinstance(name, str):
#                 load_filename = '%s_net_%s.pth' % (epoch, name)
#                 load_path = os.path.join(self.save_dir, load_filename)
#                 net = getattr(self, 'net' + name)
#                 if isinstance(net, torch.nn.DataParallel):
#                     net = net.module
#                 print('loading the model from %s' % load_path)
#                 # if you are using PyTorch newer than 0.4 (e.g., built from
#                 # GitHub source), you can remove str() on self.device
#                 state_dict = torch.load(load_path, map_location=str(self.device))
#                 if hasattr(state_dict, '_metadata'):
#                     del state_dict._metadata
#
#                 # patch InstanceNorm checkpoints prior to 0.4
#                 for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
#                     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
#                 net.load_state_dict(state_dict)
#
#     def print_networks(self, verbose):
#         """Print the total number of parameters in the network and (if verbose) network architecture
#
#         Parameters:
#             verbose (bool) -- if verbose: print the network architecture
#         """
#         print('---------- Networks initialized -------------')
#         for name in self.model_names:
#             if isinstance(name, str):
#                 net = getattr(self, 'net' + name)
#                 num_params = 0
#                 for param in net.parameters():
#                     num_params += param.numel()
#                 if verbose:
#                     print(net)
#                 print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
#         print('-----------------------------------------------')
#
#     def set_requires_grad(self, nets, requires_grad=False):
#         """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
#         Parameters:
#             nets (network list)   -- a list of networks
#             requires_grad (bool)  -- whether the networks require gradients or not
#         """
#         if not isinstance(nets, list):
#             nets = [nets]
#         for net in nets:
#             if net is not None:
#                 for param in net.parameters():
#                     param.requires_grad = requires_grad
#
#     def set_epoch(self, epoch):
#         self.current_epoch = epoch
#
#     def get_scheduler(self, optimizer, opt):
#         """Return a learning rate scheduler
#
#         Parameters:
#             optimizer          -- the optimizer of the network
#             opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
#                                   opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
#
#         For 'linear', we keep the same learning rate for the first <opt.niter> epochs
#         and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
#         For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
#         See https://pytorch.org/docs/stable/optim.html for more details.
#         """
#         if opt.lr_policy == 'linear':
#             def lambda_rule(epoch):
#                 lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
#                 return lr_l
#
#             scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#         elif opt.lr_policy == 'step':
#             scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
#         elif opt.lr_policy == 'plateau':
#             scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#         elif opt.lr_policy == 'cosine':
#             scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
#         else:
#             return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#         return scheduler


import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from torch.optim import lr_scheduler

class BaseModel(ABC):
    """这是一个模型的抽象基类（ABC）。
    要创建子类，你需要实现以下五个函数：
        -- <__init__>:                      初始化类；首先调用 BaseModel.__init__(self, opt)。
        -- <set_input>:                     从数据集中解包数据并进行预处理。
        -- <forward>:                       生成中间结果。
        -- <optimize_parameters>:           计算损失、梯度并更新网络权重。
        -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项。
    """

    def __init__(self, opt):
        """初始化 BaseModel 类。

        参数:
            opt (Option class) -- 存储所有实验标志的选项类；需要是 BaseOptions 的子类。

        在创建自定义类时，你需要实现自己的初始化。
        在这个函数中，你应该首先调用 <BaseModel.__init__(self, opt)>
        然后，你需要定义四个列表：
            -- self.loss_names (str list):          指定你想要绘制和保存的训练损失。
            -- self.model_names (str list):         指定你想要显示和保存的图像。
            -- self.visual_names (str list):        定义我们在训练中使用的网络。
            -- self.optimizers (optimizer list):    定义并初始化优化器。你可以为每个网络定义一个优化器。如果两个网络同时更新，你可以使用 itertools.chain 将它们组合在一起。参见 cycle_gan_model.py 中的示例。
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 获取设备名称：CPU 或 GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 将所有检查点保存到 save_dir
        if opt.preprocess != 'scale_width':  # 使用 [scale_width] 时，输入图像可能具有不同的大小，这会损害 cudnn.benchmark 的性能。
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # 用于学习率策略 'plateau'
        # self.current_epoch = opt.epoch_count

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的模型特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是训练阶段还是测试阶段。你可以使用此标志添加训练特定或测试特定的选项。

        返回:
            修改后的解析器。
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数:
            input (dict): 包含数据本身及其元数据信息。
        """
        pass

    @abstractmethod
    def forward(self):
        """运行前向传播；由 <optimize_parameters> 和 <test> 函数调用。"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每次训练迭代中调用"""
        pass


    def setup(self, opt):
        """
        加载网络、创建学习率调度器并打印网络结构信息。

        参数:
            opt (Option class): 包含所有实验配置的选项类，需继承自 BaseOptions。
        """
        # 如果是训练模式，创建学习率调度器
        if self.isTrain:
            # 为每个优化器创建对应的学习率调度器
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # 如果当前模式不是训练模式，或者指定了 --continue_train 参数，则加载检查点文件
        if not self.isTrain or opt.continue_train:
            # 根据 opt.load_iter 和 opt.epoch 决定加载的模型文件后缀
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            # 打印加载模型权重的日志
            print(f"[INFO] 尝试加载模型权重文件，后缀为: {load_suffix}")
            self.load_networks(load_suffix)
            print(f"[INFO] 成功加载模型权重文件，后缀为: {load_suffix}")


        # 打印网络结构信息，详细程度由 opt.verbose 决定
        self.print_networks(opt.verbose)

    def eval(self):
        """在测试时将模型设置为评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """测试时使用的前向传播函数。

        此函数将 <forward> 函数包装在 no_grad() 中，因此我们不保存中间步骤以进行反向传播。
        它还调用 <compute_visuals> 以生成额外的可视化结果。
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """计算用于 visdom 和 HTML 可视化的额外输出图像"""
        pass

    def get_image_paths(self):
        """返回用于加载当前数据的图像路径"""
        return self.image_paths

    def update_learning_rate(self):
        """更新所有网络的学习率；在每个 epoch 结束时调用"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visual_names(self):
        """返回 visual_names。test.py 将使用它来创建多预测输出目录"""
        return self.visual_names

    def get_current_visuals(self):
        """返回可视化图像。train.py 将使用 visdom 显示这些图像，并将图像保存到 HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret, self.image_paths

    def get_current_losses(self):
        """返回训练损失/错误。train.py 将在控制台上打印这些错误，并将它们保存到文件中"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name, 0.0))  # float(...) 适用于标量张量和浮点数
        return errors_ret

    def get_current_losses_tensor(self):
        """返回训练损失/错误。train.py 将在控制台上打印这些错误，并将它们保存到文件中"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name, 0.0)  # float(...) 适用于标量张量和浮点数
        return errors_ret

    def save_networks(self, epoch):
        """将所有网络保存到磁盘。

        参数:
            epoch (int) -- 当前 epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    try:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    except AttributeError as e:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复 InstanceNorm 检查点的不兼容性（0.4 之前）"""
        key = keys[i]
        if i + 1 == len(keys):  # 在末尾，指向一个参数/缓冲区
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks1(self, epoch, model_names):
        """从磁盘加载所有网络。

        参数:
            epoch (int) -- 当前 epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # 如果你使用的是 PyTorch 0.4 以上版本（例如，从 GitHub 源码构建），你可以删除 self.device 上的 str()
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 修复 0.4 之前的 InstanceNorm 检查点
                for key in list(state_dict.keys()):  # 需要在这里复制键，因为我们在循环中会改变它
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def load_networks(self, epoch):
        """从磁盘加载所有网络。

        参数:
            epoch (int) -- 当前 epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # 如果你使用的是 PyTorch 0.4 以上版本（例如，从 GitHub 源码构建），你可以删除 self.device 上的 str()
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 修复 0.4 之前的 InstanceNorm 检查点
                for key in list(state_dict.keys()):  # 需要在这里复制键，因为我们在循环中会改变它
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """打印网络中的参数总数，并在 verbose 为 True 时打印网络架构

        参数:
            verbose (bool) -- 如果为 True：打印网络架构
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """设置 requies_grad=Fasle 以避免不必要的计算
        参数:
            nets (network list)   -- 网络列表
            requires_grad (bool)  -- 网络是否需要梯度
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_scheduler(self, optimizer, opt):
        """返回一个学习率调度器

        参数:
            optimizer          -- 网络的优化器
            opt (option class) -- 存储所有实验标志的选项类；需要是 BaseOptions 的子类。
                                  opt.lr_policy 是学习率策略的名称：linear | step | plateau | cosine

        对于 'linear'，我们在前 <opt.niter> 个 epoch 保持相同的学习率，并在接下来的 <opt.niter_decay> 个 epoch 线性衰减到零。
        对于其他调度器（step、plateau 和 cosine），我们使用默认的 PyTorch 调度器。
        有关更多详细信息，请参见 https://pytorch.org/docs/stable/optim.html。
        """
        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler