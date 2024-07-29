Infrared Small Target Tracking Algorithm via Segmentation Network and Multi-strategy Fusion

R. Kou, C. Wang, Y. Yu, Z. Peng, F. Huang and Q. Fu, "Infrared Small Target Tracking Algorithm via Segmentation Network and Multi-strategy Fusion," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2023.3286836.
_____________________________________________________________________________________________________________________________________________________________________
Abstract:To solve the problem of infrared (IR) small target tracking loss or error caused by factors such as scale changes, motion blur, occlusion, etc., this paper proposes a multi-strategy fusion tracking algorithm using an IR small target segmentation network as the detection head, which mainly includes six strategies: target pixel clustering, target feature threshold adjustment, large area search, small area tracking, gate tracking, and coordinate solution. First, candidate targets are obtained through the IR small target segmentation network and pixel clustering strategy. Second, the range of candidate targets is further reduced through threshold adjustment strategy. Then, real-time tracking of IR small targets is achieved through large area search, small area tracking, and wave gate tracking strategies. Finally, the longitude, latitude, and altitude of the tracked target are obtained through coordinate calculation strategy. Both qualitative and quantitative experiments based on real IR small target sequences verify that our algorithm can achieve more satisfactory performances in terms of success rate, precision, and robustness compared with other typical visual trackers. In addition, we have deployed the tracking algorithm proposed in this study on the Orange Pi 5 embedded platform, and the tracking speed meets the real-time requirements.
___________________________________________________________________________________________________________________________________________________________________
The download address and password of the datasets (including the label):


Address：https://pan.baidu.com/s/1-NmgTdk8qKu3WozkJYnMrQ 


Password：1qaz
_____________________________________________________________________________________________________________________________________________________________________
Comparison algorithmas:

MOSSE       doi: 10.1109/CVPR.2010.5539960

CSK         doi: 10.1007/978-3-642-33765-9_50.

ECO         doi: 10.1109/CVPR.2017.733.

BACF        doi: 10.1109/ICCV.2017.129.

LADCF       doi: 10.1109/TIP.2019.2919201.

ARCF        doi: 10.1109/ICCV.2019.00298.

Auto Track  http://arxiv.org/abs/2003.12949

EFSCF       doi: 10.1016/j.neunet.2023.01.003.

In addition, we will do a job promotion: we have also compiled a set of evaluation metrics libraries suitable for infrared small target detection algorithms, named BinarySOSMetrics.

The relevant code is published on https://github.com/IRSTD/STD-EvalKit.

The main features of BinarySOSMetrics include:

High Efficiency: Multi-threading.

Device Friendly: All metrics support automatic batch accumulation.

Unified API: All metrics provide the same API, Metric.update(labels, preds) complete the accumulation of batches， Metric.get() get metrics。

Unified Computational: We use the same calculation logic and algorithms for the same type of metrics, ensuring consistency between results.

Supports multiple data formats: Supports multiple input data formats, hwc/chw/bchw/bhwc/image path, more details in ./notebook/tutorial.ipynb
