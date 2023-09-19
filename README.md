# STR🚀⚙
The project STR aims to help user train deep neural network models when the GPU memory is limited. It introduces a **s**wap dominated **t**ensor **r**e-generation strategy for training deep learning models. STR mainly contains two parts: the optimizer and the runtime built on TensorFlow. In this document, we will introduce how to use optimizer to generate the optimized strategy, and how to use a strategy in the TensorFlow runtime.

# Environment Setup
| Software   | Version |
|------------|---------|
| Ubuntu     | 20.04.2 |
| Python     | 3.6.13  |
| TensorFlow | 1.15.2  |
| Bazel      | 0.25.2  |
| CUDA       | 10.2    |
| Gurobi     | 9.1.2   |

We recommend to use Anaconda to create a virtual environment with version 3.6. The python requirenments are listed in `requirements.txt`, which can be installed through:
> pip install -r requirements.txt
> pip install -e .

# Run Experiments on TensorFlow
We implement the prototype of STR on TensorFlow. First, we need to clone TensorFlow of the version we are using:
> git clone -b r1.15 https://github.com/tensorflow/tensorflow.git \
> cd tensorflow

We provide the instrumented source files in a patch, which needs to be added to the source files of TensorFlow:
> cp STR-submit/str-tf-patch.diff tensorflow/tensorflow/ \
> cd tensorflow/tensorflow/ \
> git apply --check str-tf-patch.diff \
> git apply str-tf-patch.diff \

Afterwards, we can build a TensorFlow package with the STR runtime.
> cd /path/to/tensorflow/root/dir \
> ./configure \
> bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package \
> bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
> pip install /tmp/tensorflow_pkg/tensorflow-1.15.2-cp36-cp36m-linux_x86_64.whl

To train DNN models with optimized strategies, we need to edit a configuration for STR. Currently, we use `~/.str/` as the default configution directory. During the runtime, `~/.str/strategy.conf` will be loaded for training. For more examples of configurations, please see `./config`. The optimized solutions can be found in directory `optimizer/logs`.

Copy configurations to the target directory.
> cp -r config/* ~/.str/

After the TensorFlow is installed and the configurations are set, we can train VGG16, U-Net and MobileNet with optimized solutions. E.g., for VGG16
> cd runtime/run_vgg16 \
> nohup bash run.sh &

To train U-Net, another dataset for image segmentation is required:
> mkdir dataset; cd dataset \
> wget https://github.com/divamgupta/datasets/releases/download/seg/dataset1.zip \
> unzip dataset1.zip

# Optimizer
The optimizer is based on Gurobi, which can be downloaded from https://www.gurobi.com/. We extend a prior work Checkmate [1] to implement our swap dominated re-generation strategy, and inherit the environment it uses. The following commands can be used to optimize VGG16, U-Net and MobileNet:

> python experiments/mixed_solver_comp.py --model-name VGG16 -b 300 --ilp-eval-points 22000 \
python experiments/mixed_solver_comp.py --model-name vgg_unet -b 32 --ilp-eval-points 22000 \
python experiments/mixed_solver_comp.py --model-name MobileNet -b 400 --ilp-eval-points 22000

Then, the optimized matrices will be saved in a new directory `./data`

To speed up the optimization, the approximation can be used:
> python experiments/mixed_solver_comp_approxi.py --model-name "VGG16" -b 300 --ilp-eval-points 23000 \
python experiments/mixed_solver_comp_approxi.py --model-name "MobileNet" -b 400 --ilp-eval-points 23000 \
python experiments/mixed_solver_comp_approxi.py --model-name "vgg_unet" -b 32 --ilp-eval-points 23000


[1] Jain P, Jain A, Nrusimha A, et al. Checkmate: Breaking the memory wall with optimal tensor rematerialization[J]. Proceedings of Machine Learning and Systems, 2020, 2: 497-511.

# Citation
Please consider citing:
> @article{DBLP:journals/tpds/ZongLLWS23,
  author       = {Zan Zong and
                  Li Lin and
                  Leilei Lin and
                  Lijie Wen and
                  Yu Sun},
  title        = {{STR:} Hybrid Tensor Re-Generation to Break Memory Wall for {DNN}
                  Training},
  journal      = {{IEEE} Trans. Parallel Distributed Syst.},
  volume       = {34},
  number       = {8},
  pages        = {2403--2418},
  year         = {2023},
  url          = {https://doi.org/10.1109/TPDS.2023.3266110},
  doi          = {10.1109/TPDS.2023.3266110}
}

> @inproceedings{DBLP:conf/ipps/WenZ0L22,
  author       = {Lijie Wen and
                  Zan Zong and
                  Li Lin and
                  Leilei Lin},
  title        = {A Swap Dominated Tensor Re-Generation Strategy for Training Deep Learning
                  Models},
  booktitle    = {2022 {IEEE} International Parallel and Distributed Processing Symposium,
                  {IPDPS} 2022, Lyon, France, May 30 - June 3, 2022},
  pages        = {996--1006},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/IPDPS53621.2022.00101},
  doi          = {10.1109/IPDPS53621.2022.00101}
}
