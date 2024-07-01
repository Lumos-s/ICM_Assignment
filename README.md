# ICM_Assignment

该仓库进行了基于In-Context Matting与Robust Video Matting预测得到视频蒙版的性能对比，并定量计算出5项性能指标。

In-Context Matting链接：https://github.com/tiny-smart/in-context-matting

Robust Video Matting链接：https://github.com/PeterL1n/RobustVideoMatting

由于RVM的预测可以使用作者准备的colab在线进行，故本仓库的主要内容及环境配置遵循ICM所需的环境配置。

## 环境配置

本仓库遵循ICM所需的环境配置。具体的环境安装过程参考https://github.com/tiny-smart/in-context-matting

补充：ICM需要在linux环境下运行，且GPU显存至少24G，推荐使用python==3.10。此项未在原文中提出。

## 数据

下载来源：https://pan.baidu.com/s/11agdVdfsVM5r3PsaX626mg?pwd=sjrz

提取码：sjrz

说明：image文件夹内为视频帧图片，alpha文件夹内为手动标注的蒙版，trimap文件夹内为生成的trimap图像。

下载完成后，解压至数据目录，覆盖原有datasets文件夹即可。

## 代码运行

- trimap的生成

将tri.py移动至 ./datasets/HP 目录下，在该目录下启动终端，运行该文件即可自动生成trimap。

```
python tri.py
```

- 使用ICM模型进行预测

预训练模型的获取参考上文环境中给出的链接。

```
python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
```

- 评估指标的计算

将 ./datasets/HP 目录下的 alpha 文件夹与预测结果 ./results 文件夹均移动至 ./metric 目录下，并将 results 文件夹重命名为 icm 。

在 metrics 目录下，运行 metrics.py 文件即可计算出对于ICM的评估指标。

```
python metrics.py
```

若要运行RVM或其他模型的评估指标，请在 metric 目录下新建相应的文件夹，在此假设命名为 MY_RESULTS ，将生成的图像放入该文件夹内，同时对 metrics.py 进行如下修改：

第157行：

```
predicted_dir = "MY_RESULTS/"  # 更改数据路径
```

第163行：

```
ground_truth_frames = load_images(ground_truth_dir, is_ground_truth=False)  # 更改为False
```

注意：为了使结果易于观察，指标中MAD与MSE分别乘了1000，dtSSD乘了100。

## 实验

作者进行的实验以及结果详见report.pdf。

如果有遗漏或者不解之处，请随时联系我 yusongren@hust.edu.cn