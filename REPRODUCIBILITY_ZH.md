# 复现入口（内部 staging）

正式实验应使用干净的环境、明确的数据根目录和独立的输出目录。不要把服务器绝对路径写进公开脚本。

核心流程应包含：

1. 安装经过审计的公开依赖；
2. 按 `datasets/` 中的 loader 和退化实现准备 CAVE 或 Harvard；
3. 用 `Train_Cave.py` 或 `Train_Harvard.py` 从随机初始化训练；
4. 只按规定的验证条件选择 checkpoint；
5. 使用统一评测工具报告 PSNR、SAM、ERGAS 和 SSIM；
6. 在干净机器上运行 smoke test，确认模型注册、CUDA 扩展和任意输出尺寸接口。

当前 staging 只保存代码，不保存训练权重和数据。具体 checkpoint、实验配置和历史结果仍在私有研究归档中。
