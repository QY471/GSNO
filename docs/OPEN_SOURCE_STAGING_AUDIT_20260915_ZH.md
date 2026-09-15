# GSFusion 开源 staging 审计记录

日期：2026-09-15

## 结论

本次建立的是不公开的发布 staging，不是最终公开仓库。原研究项目、完整实验资产和退役备份均保持不动。

## staging 位置

- 194：`/mnt/16t-2/sqy/GSFusion_public_staging_20260915`
- 本地：`D:\asus\桌面的东西\科研存放\GSFusion_public_staging_20260915`

## 已纳入

- `Train_Cave.py`、`Train_Harvard.py`；
- `model/`、`datasets/` 数据加载与退化代码；
- `ops/msi_conditioned_gaussian_renderer/`、`extensions/`、`submodules/`；
- 选定的 smoke、audit、evaluation 和指标工具；
- staging README、复现说明、排除范围和第三方来源待审计清单。

## 已排除

- 原始/处理后数据集与 LSO 真实数据结果；
- 所有 checkpoint、训练日志、analysis 产物和服务器队列脚本；
- `external_baselines/` 第三方 baseline 源码；
- 服务器绝对路径和 `sqy` 环境路径（`Train_Cave.py` 的 staging 副本已移除固定服务器回退路径）。

## 已验证

- staging 不含 `Checkpoint*`、`analysis`、`external_baselines`、原始数据文件、权重和日志；
- staging 中没有 `/mnt/`、`/home/star`、Windows 本机路径等路径引用；
- 在 `sqy` 环境中 `compileall` 返回 `0`；
- 原项目没有被移动、删除或覆盖。

## 尚未解除的发布门槛

1. 选择根目录开源许可证；
2. 核对 CUDA 扩展、submodule 和 Python 依赖的上游许可证；
3. 把服务器 requirements 拆成公开安装依赖与内部锁定环境；
4. 补齐 CAVE/Harvard 数据下载、预处理和目录说明；
5. 在干净机器上完成一次从零安装、smoke test、训练和评测；
6. 决定哪些 checkpoint 可以作为独立 release asset 发布。

在上述门槛完成前，不应创建公开 GitHub 仓库或公开 release。
