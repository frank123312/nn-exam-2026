# 一键运行与 LaTeX 报告说明

## 1. 安装依赖

```bash
/usr/local/bin/python3 -m pip install numpy scipy scikit-learn matplotlib pillow torch
```

第 6 题脚本现在直接读取 MNIST 的 IDX 文件，不再依赖 `torchvision` 下载器。MNIST 数据应放在：

```text
mnist_data/MNIST/raw/
```

目录中至少应包含以下四个解压后的标准 IDX 文件：

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

如果只有 `.gz` 压缩文件，脚本会自动解压。

## 2. 一键运行全部题目

在项目目录执行：

```bash
/usr/local/bin/python3 run_all.py
```

脚本会运行第 2、4、5、6、7 题，并生成：

```text
generated/logs/q2.log
generated/logs/q4.log
generated/logs/q5.log
generated/logs/q6.log
generated/logs/q7.log
generated/report_values.tex
```

图片和 CSV 会保存在各题对应目录中。

`report.tex` 会自动读取 `generated/report_values.tex`。因此，第 6 题的多随机种子结果不需要手动复制粘贴：运行 `run_all.py` 后，报告中的数字会自动更新。

## 3. 常用运行方式

跳过训练耗时较长的第 6 题，快速检查其他题目：

```bash
/usr/local/bin/python3 run_all.py --skip-q6
```

只运行第 6 题，并自动刷新报告数值：

```bash
/usr/local/bin/python3 run_all.py --q6-only
```

如果已经单独运行过某个脚本，只想根据现有日志和 CSV 重新生成 LaTeX 数值宏：

```bash
/usr/local/bin/python3 run_all.py --refresh-values-only
```

## 4. 编译 LaTeX 报告

建议安装 MacTeX 或 BasicTeX，并使用 XeLaTeX：

```bash
xelatex report.tex
xelatex report.tex
```

生成文件：

```text
report.pdf
```

## 5. 第 4 题已经按原始试卷调整

代码和报告现在使用题目中的高斯调谐函数：

```text
f_a(s) = exp(-(s - s_a)^2 / (2 sigma_a^2)),  s in [0, 180 degrees]
```

当前数值模拟参数在 `q4_direction_decoder_template.py` 顶部明确给出，可以在报告中说明。

## 6. 提交前检查

1. 运行完整的一键脚本，确保第 6 题表格中的 `--` 被多随机种子实验结果替换。
2. 编译两次 `report.tex`。
3. 检查姓名、学号、图表和试听结论。
4. 按老师要求，将填写姓名和学号后的试题页放在回答页之前。

## Q7 interactive endpoint selector

The automatic batch script uses a fixed reachable endpoint so that all experiments are reproducible. An optional interactive interface is also provided:

```bash
python3 q7_maze_interactive.py
```

A matplotlib window will open. Click a reachable dark corridor cell as the goal. The script will train Q-learning for the selected endpoint, print the BFS shortest-path length and the Q-learning path length, and draw the route.

For command-line use without clicking:

```bash
python3 q7_maze_interactive.py --goal 8 45
```
