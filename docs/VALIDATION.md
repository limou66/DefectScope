# 交付验证

- 本机 Python 3.14 / Windows，11 项测试通过，包含真实 ResNet18 模型推理。
- 轻量核心测试验证 AUROC ties、conformal 排序边界、最近邻、coreset、数据隔离、保存加载一致性、光照拟合、HTTP 输入与上传。
- 真实数据：MVTec AD bottle，三个种子，四策略；另有种子 7 光照开关消融。
- 合成数据：三个类别、三个种子、光照开关、四策略，共 72 组指标。
- 浏览器实际检查样本加载、破损图像检测、热力图、掩码与证据；修复了相对样本路径和 hidden 元素 CSS 问题。
- 构建 Python wheel 成功。
- GitHub Actions 配置已准备，本地尚未登录 GitHub，远程 CI 尚未执行。

测试命令：DEFECTSCOPE_TEST_DEEP=1 时运行 python -m pytest -q；无真实数据/权重时核心测试通过，深度集成测试跳过。深度依赖和模型的首次下载需网络。
