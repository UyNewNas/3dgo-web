# 3D Go AI Training

这个仓库包含一个使用 Playwright 与浏览器交互的 3D 围棋训练脚本 `train_go_ai.py`。脚本使用 PyTorch 实现一个简单的 DQN。下面说明如何启用显卡训练（CUDA）。

## 要求

- Python 3.8+
- PyTorch（如果要使用 GPU，请安装带 CUDA 支持的版本）
- playwright
- gymnasium

示例安装（CPU 版 PyTorch，仅作参考，实际请参考 PyTorch 官网安装命令）：

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install playwright gymnasium numpy
python -m playwright install

如果要使用 GPU（CUDA），请按你系统和 CUDA 版本从 https://pytorch.org/ 获取相应的安装命令。

## 运行说明

有三种方法选择运行设备：

1. 自动选择（默认）: 如果系统可用 CUDA，会自动使用 `cuda`，否则使用 `cpu`。

2. 通过环境变量指定：

在 Windows cmd 中：

```cmd
set TRAIN_DEVICE=cuda
python train_go_ai.py
```

3. 通过命令行参数指定：

```cmd
python train_go_ai.py --device cuda
```

注意：要在 GPU 上训练，请确保已正确安装支持 CUDA 的 PyTorch，并且显卡驱动与 CUDA 版本匹配。

## 其他

- 训练过程中模型会定期保存为 `go_ai_{episode}.pth`。
- 如果你仅想尝试脚本，可保持 `--device cpu` 来避免显卡依赖。

## 非法动作（已有棋子 / 禁着点）处理

训练脚本已经集成页面中暴露的判断逻辑：

- 当智能体选择的坐标位置 `isPositionOccupied(x,y,z)` 为 true（页面上已有棋子）时，环境会把该动作视为非法，返回当前状态、奖励为 -1（惩罚），并且不会结束当前局（done=False）。
- 当智能体选择的坐标位置 `isForbiddenPoint(x,y,z,color)` 为 true（自杀或打劫等禁着点）时，环境会把该动作视为非法，返回当前状态、奖励为 -2（更强烈惩罚），并且不会结束当前局（done=False）。

这些设计使训练时智能体可以学习避免无效或规则不允许的落子。你可以根据需要修改这些惩罚值或改为直接结束当前局（把 done 设为 True）。

建议：开始训练时先用较小的惩罚（例如 -1），观察智能体是否能学会避免无效动作，再根据训练表现调整惩罚强度或采取更严格的策略。

