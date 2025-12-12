import torch
import torch.nn as nn

# ==========================================================================================
# ControlInjectionBlock - 最终完整实现
# ==========================================================================================

class ControlInjectionBlock(nn.Module):
    """
    一个实现了“终极方案”的控制注入模块。
    它在单个UNet注入点，融合了三大核心技术：
    1. 自适应实例归一化 (AdaIN): 对齐控制信号的风格与UNet特征的风格。
    2. 动态FiLM调制: 根据时间步生成逐通道的alpha和beta，实现时间感知的强度调制。
    3. 空间门控: 根据UNet特征内容生成空间掩码，实现精确的局部控制。
    """
    def __init__(self, channels: int, time_emb_dim: int, hidden_dim: int = 512):
        """
        初始化函数。
        参数:
            channels (int): 当前UNet层特征图的通道数 (例如: 320, 640, 1280)。
            time_emb_dim (int): Stable Diffusion时间步嵌入的维度 (通常为 1280)。
            hidden_dim (int): FiLM调制网络中间层的维度。
        """
        super().__init__()
        self.channels = channels

        # 1. 动态FiLM调制网络 (Time-step Modulation MLP)
        #    - 角色: 时间-通道 调制器
        #    - 任务: 根据时间步 emb，生成逐通道的 alpha 和 beta。
        self.film_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * channels) # 输出2*C维向量，分别用于alpha和beta
        )

        # 2. 空间门控网络 (Spatial Gating Network)
        #    - 角色: 空间 选择器
        #    - 任务: 根据UNet的中间特征 h，生成空间注意力掩码 gate。
        self.gate_conv = nn.Sequential(
            # 使用3x3卷积以感知局部上下文，并引入瓶颈结构减少参数
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            # 使用1x1卷积将通道数压缩到1，生成最终掩码
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            # 使用Sigmoid将输出约束在0~1之间，使其成为有效的概率掩码
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，执行三阶段的精确注入流程。
        参数:
            h (torch.Tensor): UNet的中间特征图，形状为 (B, C, H, W)。
            emb (torch.Tensor): 时间步嵌入，形状为 (B, time_emb_dim)。
            controls (torch.Tensor): 来自Adapter编码器的控制信号，形状为 (B, C, H, W)。
        返回:
            torch.Tensor: 经过控制注入后更新的特征图。
        """
        # --- 步骤 0: 自适应实例归一化 (AdaIN) ---
        # 目标: 将controls的内容（结构）与h的风格（纹理）对齐。
        mean_h, std_h = torch.mean(h, dim=(1,2, 3), keepdim=True), torch.std(h, dim=(1,2, 3), keepdim=True)
        mean_control, std_control = torch.mean(controls, dim=(1,2, 3), keepdim=True), torch.std(controls, dim=(1,2, 3), keepdim=True)

        # 添加一个小的epsilon或使用torch.where来防止除以零的错误
        std_control = torch.where(std_control < 1e-6, torch.ones_like(std_control), std_control)

        aligned_controls = (controls - mean_control) / std_control * std_h + mean_h

        # --- 阶段一: 准备“弹药” (时间-通道调制) ---
        # 根据时间步emb，动态生成逐通道的FiLM参数
        film_params = self.film_mlp(emb)
        
        # 将输出切分为alpha_base和beta
        alpha_base, beta_t = torch.chunk(film_params, 2, dim=-1)
        
        # 关键技巧: alpha = alpha_base + 1，确保初始化时强度为1，加速稳定收敛
        # 调整形状为 (B, C, 1, 1) 以便进行广播操作
        alpha_t = alpha_base.view(-1, self.channels, 1, 1) + 1
        beta_t = beta_t.view(-1, self.channels, 1, 1)

        # 对已对齐的控制信号进行完整的FiLM调制
        modulated_controls = aligned_controls * alpha_t + beta_t
        # modulated_controls = controls * alpha_t + beta_t

        # --- 阶段二: 锁定“目标” (空间选择) ---
        # 根据UNet的当前内容h，生成空间掩码
        gate = self.gate_conv(h) # 输出形状为 (B, 1, H, W)

        # --- 阶段三: 精确“打击” (最终融合) ---
        # 将调制好的控制信号，通过空间门控，只注入到需要它的地方
        h_out = h + gate * modulated_controls
        
        return h_out

# ==========================================================================================
# 运行示例 (用于独立测试)
# ==========================================================================================
if __name__ == '__main__':
    # --- 模拟输入数据 ---
    batch_size = 4
    channels = 320      # 模拟UNet浅层的通道数
    height, width = 64, 64 # 模拟UNet浅层的特征图尺寸
    time_emb_dim = 1280 # Stable Diffusion中time_emb的维度

    # 模拟UNet的中间特征 h
    h_feature = torch.randn(batch_size, channels, height, width).cuda()
    
    # 模拟时间步嵌入 emb
    time_embedding = torch.randn(batch_size, time_emb_dim).cuda()
    
    # 模拟来自Adapter的控制信号 controls
    control_feature = torch.randn(batch_size, channels, height, width).cuda()

    # --- 初始化并运行模块 ---
    print("初始化 ControlInjectionBlock...")
    injection_block = ControlInjectionBlock(
        channels=channels, 
        time_emb_dim=time_emb_dim
    ).cuda()
    
    print("模块参数量: {:.2f}M".format(sum(p.numel() for p in injection_block.parameters()) / 1e6))

    print("\n执行前向传播...")
    output_feature = injection_block(h_feature, time_embedding, control_feature)
    
    print("\n前向传播完成！")
    print(f"输入特征图形状: {h_feature.shape}")
    print(f"输出特征图形状: {output_feature.shape}")
    
    # 验证输出形状是否正确
    assert output_feature.shape == h_feature.shape
    print("\n输出形状验证通过。")