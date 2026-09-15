# GSFusion 重要模型代码导览

新模型禁止继续直接堆在`model/`根目录。代码分类、命名、注册、日志和checkpoint落盘的强制规则见`docs/runbooks/GSFUSION_EXPERIMENT_OUTPUT_AND_MODEL_PLACEMENT_STANDARD_ZH.md`。当前仍位于根目录的活动模型属于路径兼容例外，关联训练和评测结束前不移动。

这个目录不是“十几个完全不同的模型”。大部分 `HRFused` 文件共享同一套主干，只是在控制实验中替换一个局部环节。阅读时先看家族关系，不要逐个文件从头理解。

## 1. 目前最重要的主线

当前研究对象是：

```text
LR-HSI → HSI encoder → bicubic到HR → F_H ─┐
                                          ├→ concat → fusion → F_fused ─────┐
HR-MSI → MSI encoder              → F_M ─┘                                 │
                                                                               ├→ decoder → 31波段残差
某种primitive特征 → Gaussian heads → normalized adaptive-3σ scatter → delta ─┘

prediction = bicubic(raw LR-HSI) + 31波段残差
```

这里Gaussian不负责LR→HR上采样，也不直接输出最终31波段。它在HR latent网格运输一份局部修正 `delta`。

当前最重要的两个已完成模型：

1. `GSFusion_HRFused_Circular_PrimitiveEmbedding.py`：E3，专用primitive embedding，固定中心圆Gaussian。
2. `GSFusion_E6_MSIRoutedGaussian.py`：E6，在E3基础上用MSI引导选择邻近HSI内容，目前综合最好。

正在验证的两个后续方向：

1. `GSFusion_HRFused_FullGeometry_PrimitiveEmbedding.py`：E3保持不变，只恢复完整geometry。
2. `GSFusion_E6_LRPrimitiveGaussianTransport.py`：把primitive从HR像素改到LR-HSI cell，再运输到HR。

## 2. HRFused家族为什么看起来都差不多

它们共同保留：

- HSI/MSI两路1×1浅层投影；
- 两路各3层ADCI；
- `F_H`与`F_M`在HR网格concat；
- 1×1 fusion与1×1 decoder；
- raw LR-HSI bicubic base；
- 一次CUDA Gaussian residual；
- density normalization；
- adaptive 3σ窗口；
- Gaussian residual零初始化。

真正变化的只有以下三类问题：

| 问题 | 可选设计 |
|---|---|
| Gaussian读取什么 | `F_fused`、独立`E_g`、加入HSI旁路、加入routing |
| geometry学什么 | 圆std；或offset、rho、std_x/std_y完整geometry |
| Gaussian之后做什么 | 直接decoder；或再加轻量HR局部卷积 |

因此这些文件看起来有大量重复是正常的：它们最初是为了保证每次只改一个变量，避免实验之间互相污染。

## 3. HRFused逐文件说明

### 3.1 `GSFusion_HRFused_AdaptiveGaussianResidual.py`

第一版表现较强的新版HR-Fused Gaussian。

```text
F_H + F_M → fusion → F_fused
F_fused → value + opacity + offset + std_x/std_y + rho
→ full-geometry Gaussian delta
F_fused + delta → decoder
```

- geometry：完整geometry；
- primitive来源：普通融合特征`F_fused`；
- 4×历史最好：约52.5671 dB；
- 用途：证明新版normalized adaptive-3σ单残差设计明显优于旧Scatter；也是后续geometry实验的历史起点。

### 3.2 `GSFusion_HRFused_Circular_PrimitiveEmbedding.py`——E3

当前核心基线之一。

```text
J = concat(F_H,F_M)
J → fusion → F_fused                 重建路径
J → 1×1残差embedding → E_g          primitive路径
E_g → value + scalar std + opacity
→ circular Gaussian delta
F_fused + delta → decoder
```

- offset固定0；rho固定0；`std_x=std_y`；
- 重建特征与Gaussian primitive特征分开；
- 4×/8×/16×/32×：52.5337/50.5115/47.4501/44.4328；
- 用途：研究专用primitive embedding及尺度稳定性。

### 3.3 `GSFusion_E6_MSIRoutedGaussian.py`——E6正式入口

当前综合结果最好的模型。它保留E3公共骨架，只改变Gaussian value来源。

```text
F_H提供query和3×3邻域HSI value
F_M提供key
→ MSI引导组合附近HSI内容
→ routing_delta

value_source = E_g + routing_delta
geometry_source = E_g
→ circular Gaussian delta
→ F_fused + delta → decoder
```

- geometry仍是固定中心圆Gaussian；
- routing影响“运输什么”，不预测offset或椭圆方向；
- 4×/8×/16×/32×：52.5737/50.5382/47.5060/44.4382；
- 正式训练注册名：`e6_msi_routed_gaussian`。

### 3.4 `GSFusion_HRFused_Circular_PrimitiveValueCommon.py`

这不是独立实验模型，而是E5/E6/E7共用的代码组件，包含：

- ADCI双路骨架；
- `F_fused`和`E_g`生成；
- 圆Gaussian renderer；
- MSI引导3×3 HSI routing；
- primitive统计。

`GSFusion_E6_MSIRoutedGaussian.py`依赖它，不能单独删除或移动。

### 3.5 `GSFusion_E6_Combined_Annotated.py`

E6的单文件中文注释学习版，将正式E6入口和Common代码合在一起。

- 数学和正式E6一致；
- 已用E6 best checkpoint验证输出逐元素一致；
- 用途：阅读、交接和审计；
- 正式训练仍建议使用`GSFusion_E6_MSIRoutedGaussian.py`，避免维护两份实现。

### 3.6 `GSFusion_HRFused_FullGeometry_PrimitiveEmbedding.py`

正在运行的严格E3 geometry对照。

```text
其他全部等于E3
圆geometry → opacity + offset_x/y + std_x/y + rho
```

- 与E3共享78个epoch-0张量；
- 初始opacity和std完全等于E3；
- 新增offset/rho从0开始；
- 只多260个参数；
- 用途：回答E3泛化好是因为primitive embedding，还是固定圆geometry也提供了关键约束。

### 3.7 `GSFusion_E6_LRPrimitiveGaussianTransport.py`

另一条更激进的研究分支，不属于普通HRFused残差。

```text
LR-HSI cell产生primitive
HR-MSI提供HR footprint/引导
LR primitive直接Gaussian transport到HR网格
```

- Gaussian真正承担LR cell到HR网格的跨尺度运输；
- 与E3/E6的“已经在HR网格上做局部修正”不同；
- 当前属于正在验证的架构实验。

### 3.8 `GSFusion_HRFused_Circular_E6GaussianAwareHRLocalReconstruction.py`——E8

在E6的Gaussian注入之后增加极轻量HR局部重建：

```text
F_fused + Gaussian delta → refined
refined → depthwise 3×3 + 1×1 → local_delta
refined + local_delta → 原decoder
```

- local输出零初始化，训练起点等价于E6；
- 用途：针对4×边缘和高频误差；
- 它测试的是“Gaussian之后是否还需要普通局部卷积”，不是新的Gaussian设计。

## 4. EDSR三模型是一组独立对照

### `GSFusion_EDSRBackbone_NoGaussian.py`

纯EDSR卷积骨架，没有Gaussian。用于判断性能究竟来自backbone还是Gaussian。

### `GSFusion_EDSRBackbone_CircularGaussianResidual.py`

EDSR深层特征直接预测Gaussian。4×较好，但跨倍率明显下降，说明Gaussian直接读取高度适应4×的卷积特征可能传播错误内容。

### `GSFusion_EDSRBackbone_PrimitiveEmbedding_CircularGaussianResidual.py`

EDSR负责重建，另设1×1 primitive embedding给Gaussian。它与Direct的4×接近，但16×/32×明显更稳定。

| EDSR模型 | 4× | 8× | 16× | 32× |
|---|---:|---:|---:|---:|
| NoGaussian | 52.5169 | 49.6242 | 43.8779 | 38.5927 |
| Gaussian Direct | 52.6068 | 49.0596 | 42.4238 | 37.2011 |
| Primitive Embedding Gaussian | 52.6028 | 49.9182 | 45.8094 | 41.8811 |

`EDSR.py`只是这些模型使用的EDSR基础模块，不是一个完整HSI融合训练入口。

## 5. GSNO与旧Scatter参考组

### `GSFusion_GSNO.py`

原GSNO gather版本：融合后的HR特征经过3个局部Gaussian gather/FFN块。它不是CUDA scatter，Gaussian也不承担LR→HR上采样。

### `GSFusion_GSNO_NoGS_Identity.py`

删除GSNO Gaussian gather后的强骨架对照，用于估计旧gather分支的独立贡献。

### `GSFusion_GSNOBackbone_ScatterGS.py`

在同一个GSNO强骨架中插入3个旧v2 CUDA Scatter层。它用于说明旧Scatter设计为何慢、且效果不如新版单残差。

### `GSFusion_Baseline.py`

最早期EDSR特征提取加3层旧CUDA Gaussian Scatter的历史baseline。保留它是为了复现早期实现，不建议作为当前新实验起点。

## 6. 另一条Gaussian上采样路线

### `GSFusion_MSI_Guided_ScaleConsistent.py`

该文件与HRFused家族的职责完全不同：

```text
LR-HSI latent
→ MSI指导Gaussian上采样
→ HR-HSI latent
→ 与HR-MSI特征融合
→ decoder
```

这里Gaussian位于LR→HR上采样位置；HRFused E3/E6则先bicubic得到`F_H`，Gaussian只处理HR网格上的残差。两条路线不要混称。

## 7. 阅读与使用建议

如果只想理解当前主线，按下面顺序阅读即可：

1. `GSFusion_HRFused_Circular_PrimitiveEmbedding.py`：理解E3；
2. `GSFusion_HRFused_Circular_PrimitiveValueCommon.py`：理解公共骨架与routing；
3. `GSFusion_E6_MSIRoutedGaussian.py`：理解E6新增的几行逻辑；
4. `GSFusion_HRFused_FullGeometry_PrimitiveEmbedding.py`：理解当前geometry对照；
5. `GSFusion_E6_LRPrimitiveGaussianTransport.py`：最后再看LR primitive新路线。

如果只想正式训练当前已验证模型：

- E3使用`hr_fused_circular_primitive_embedding`；
- E6使用`e6_msi_routed_gaussian`。

不要把以下文件误当成独立正式模型：

- `EDSR.py`：基础网络组件；
- `GSFusion_HRFused_Circular_PrimitiveValueCommon.py`：E6公共组件；
- `GSFusion_E6_Combined_Annotated.py`：E6阅读/审计副本。

## 8. 目录整理原则

- 当前重要、可复现、正在训练或需要直接对照的入口保留在`model/`根目录；
- 必需但不直接训练的共享实现放在`important_model_support/`；
- 已结束且研究信号较弱的E1/E2/E4/E5/E7等保存在`relatively_less_important_models/`；
- 没有删除历史代码；训练日志和checkpoint按同样原则分别保留在主Checkpoint或相对次要目录。

## 9. 2026-08-05第一阶段目录整理

已完成首轮非活动代码整理。当前模型包按职责分为：

```text
model/
├── archive/annotated/     # 阅读和注释副本
├── archive/legacy/        # 已降级但保留的历史模型
├── baselines/             # GSNO旧对照、EDSR对照和最早期baseline
├── geometry/              # 旧自适应几何及EDSR使用的各向同性组件
├── support/               # EDSR基础组件
└── transport/             # 未注册的半稠密评测入口
```

本轮保持原位、禁止移动的活动依赖包括：

- `GSFusion_E3_ParallelConstrainedEllipticalGaussian.py`；
- `GSFusion_E3_ParallelTwoConstrainedEllipticalGaussian.py`；
- `GSFusion_E3_ThreeConstrainedEllipticalGaussian.py`；
- `GSFusion_E3_ConstrainedEllipticalGaussian.py`；
- `GSFusion_E3_GaussianMechanismAblationCommon.py`；
- `GSFusion_E3_HRFactorizedReferenceContinuous.py`及其连续参考依赖；
- `GSFusion_HRFused_Circular_PrimitiveEmbedding.py`和`GSFusion_GSNO.py`。

`Train_Cave.py`的非活动注册项已经指向新包路径，历史模块仍可通过原模型键加载。迁移前后的SHA清单和活动进程记录位于`docs/maintenance/MODEL_DIRECTORY_CLEANUP_20260805/`。
