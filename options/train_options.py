from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    该类包含训练相关的选项，同时继承并共享 BaseOptions 中的通用选项。
    """

    def initialize(self, parser):
        # 调用父类 BaseOptions 的 initialize 方法，初始化解析器
        parser = BaseOptions.initialize(self, parser)

        # visdom 和 HTML 可视化相关参数
        parser.add_argument('--display_freq', type=int, default=400,
                            help='显示训练结果到屏幕的频率（单位：迭代次数）')
        parser.add_argument('--display_ncols', type=int, default=5,
                            help='如果值为正，则在 visdom 面板中按指定列数显示图像')
        parser.add_argument('--display_id', type=int, default=0,
                            help='visdom 窗口 ID，用于区分不同显示窗口')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom 服务器地址')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom 显示的环境名称（默认值为 "main"）')
        parser.add_argument('--display_port', type=int, default=8097,
                            help='visdom 服务器的端口号')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='将训练结果保存到 HTML 的频率（单位：迭代次数）')
        parser.add_argument('--print_freq', type=int, default=800,
                            help='在控制台打印训练结果的频率（单位：迭代次数）')
        parser.add_argument('--no_html', action='store_true',
                            help='如果设置，则不保存训练结果到 [opt.checkpoints_dir]/[opt.name]/web/')

        # 网络模型的保存和加载相关参数
        parser.add_argument('--save_epoch_freq', type=int, default=25,
                            help='每隔指定 epoch 保存一次模型检查点的频率')
        parser.add_argument('--save_by_iter', action='store_true',
                            help='是否按迭代次数保存模型检查点')
        parser.add_argument('--continue_train', action='store_true',
                            help='继续训练：加载最近保存的模型')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='训练的起始 epoch 编号')
        parser.add_argument('--phase', type=str, default='train',
                            help='设置当前运行的阶段，例如 train、val、test 等')

        # 训练相关参数
        parser.add_argument('--niter', type=int, default=75,
                            help='保持初始学习率的 epoch 数量')
        parser.add_argument('--niter_decay', type=int, default=75,
                            help='学习率线性衰减到 0 所需的 epoch 数量')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Adam 优化器的动量项')
        parser.add_argument('--lr', type=float, default=0.0005,
                            help='Adam 优化器的初始学习率')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='学习率调整策略：[linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='每隔指定迭代次数乘以一个衰减因子 gamma')

        # 设置训练模式标志
        self.isTrain = True

        # 返回更新后的解析器
        return parser

