# 原创实现与参考边界

本项目没有克隆或复制其他异常检测开源仓库。代码由本次项目设计与开发完成，使用现成数值库、图像库与官方预训练模型。AI 辅助生成不自动等于个人已掌握；提交简历前应按 LEARNING_GUIDE.md 复现并完成自己的实质改进。

| 部分 | 来源 / 性质 | 可合理表述 |
|---|---|---|
| patch 记忆库异常检测思想 | 已有方法；参考 PatchCore | 参考正常特征记忆库思路，独立实现 |
| ResNet18 结构与权重 | torchvision / ImageNet | 使用冻结预训练视觉特征 |
| 贪心最远点选择 | 经典选择算法 | 实现有界候选池的记忆压缩 |
| 空间均值 / 对角标准差 | 经典统计异常检测 | 实现逐位置正常统计分支 |
| 分支归一化与空间可靠性权重 | 本项目组合设计 | 设计并消融空间可靠性融合策略 |
| IRLS 光照平面与模板组合 | 鲁棒回归已有，本项目应用实现 | 实现并验证可关闭的光照干扰校正 |
| conformal p 值 | 已有统计方法 | 实现独立正常样本排序校准 |
| 合成数据、实验脚本、API、网页、测试 | 本项目实现 | 构建可复现算法实验与演示系统 |

没有进行覆盖全部相关工作的系统查新。因此不可使用“首次提出”“全新算法”“全球首创”等研究原创性措辞，也不能把表中组合设计写成已证明的 SOTA。

## 与 PatchCore 的差异

该实现不是 PatchCore 复现。没有使用论文的 WideResNet50、论文相同特征池化与重加权、论文评测协议或完整类别平均。项目 global 模式只是我们自己的记忆库基线。门控及光照方法的性能必须与本项目控制变量实验一起解读。

## 参考

1. Roth et al., Towards Total Recall in Industrial Anomaly Detection, CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
2. He et al., Deep Residual Learning for Image Recognition, CVPR 2016. https://arxiv.org/abs/1512.03385
3. Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. https://arxiv.org/abs/2107.07511
4. MVTec AD official dataset and license. https://www.mvtec.com/research-teaching/datasets/mvtec-ad
5. Torchvision official ResNet18 documentation. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
