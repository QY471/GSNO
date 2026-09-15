# GSNO 本地可迁移档案索引

日期：2026-09-15

这个目录是可直接继续开发的公开代码工作仓库，不依赖 194 服务器，也没有配置外部远程地址。

## 本地内容

- 当前代码仓库：本目录；
- Git 初始提交：`67db616`；
- 原项目完整 Git 历史：相邻目录 `GSFusion_Archive_20260915/GSFusion_git_current_branch_20260915.bundle`；
- 完整项目归档：相邻目录 `GSFusion_Archive_20260915/GSNO_PROJECT_FULL_20260915.tar`；
- 数据集归档：相邻目录 `GSFusion_Archive_20260915/GSNO_DATASETS_20260915.tar`。

## 说明

公开代码仓库只保留源码、必要配置、CUDA 扩展源码、数据加载/退化逻辑、评测工具和复现说明。数据集、checkpoint、日志和第三方 baseline 不放进这个 Git 仓库，避免误发布和仓库膨胀。

原始 bundle 保留完整历史对象；由于原项目含有 Linux 下的特殊文件名和超长路径，不能在 Windows 上完整 checkout，这不影响 bundle 本身作为历史归档使用。

正式公开前仍需完成许可证、第三方代码来源、数据下载说明和干净环境复现审计。
