# TED调试笔记

TED SFD 等等等kitti benchmark里的全都是魔改Voxel RCNN

其实都是魔改Pointnet

因为调试报错都完全一样。。。。。服了



## 前言

本机运行环境比较新，所以对源码进行一定微调，且数据集不方便转移到系统盘，故更改路径使得能够在数据盘读取到kitti

ubuntu=20.04

cuda=11.7

pytorch=1.13

python=3.8.15

显卡：rtx3080ti(vram=12G)





## 编译

从报错内容不难看出，该文章时2021年AAAI的Voxel RCNN的魔改

根据开源代码的运行指令

```sh
python3 setup.py develop
```

会报错

```sh
Traceback (most recent call last):
  File "setup.py", line 5, in <module>
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
ModuleNotFoundError: No module named 'torch'
```

是因为pytorch配置在了conda环境中，使用python3与sudo python运行的是本地解释器，故应当运行

```sh
python setup.py develop
```



这里直接运行报错

```sh
cc1plus: fatal error: /usr/local/cuda/include/stdc-predef.h: 权限不够
```

**但是 压根没有这个头文件**



尝试解决，加库

```sh
sudo apt-get install g++-multilib
```

没用

加库



3.2update

看了issue

应当先运行

```sh
pip3 install -r requirements.txt
```

一来一键安装各种库，二来避免冲突



其实stdc-predef.h的位置在/usr/include/stdc-predef.h

但是cuda创建软链接没有给链接就很怪，不妨手动cp一下或者重新创建软链

手动cp过后其实存在文件了，但是还是有权限不够的问题



检查权限后确实是没有全局读写的权限

```
sudo ls -l /usr/local/cuda/include/stdc-predef.h
[sudo] bopang 的密码： 
-rw-r--r-- 1 root root 2290 4月   7  2022 /usr/local/cuda/include/stdc-predef.h
```

这里其实是/usr/local/cuda/include的权限太高

使用

```
sudo chmod 777 /usr/local/cuda/include
```

**成功解决！**





报错

```sh
/home/bopang/lidar_object_detection/TED/pcdet/ops/votr_ops/src/build_mapping.cpp:8:10: fatal error: THC/THC.h: 没有那个文件或目录
 #include <THC/THC.h>
```

解决

根据issuehttps://github.com/hailanyi/TED/issues/4

其原因是pytorch版本太高https://github.com/open-mmlab/mmdetection3d/issues/1332#issuecomment-1085991179

pytorch1.11中删除了这个库，需要降低pytorch版本

其实也可以直接改接口

参考如下

https://blog.csdn.net/qq_36891089/article/details/124353149

首先注释掉

```c++
#include <THC/THC.h>
```

然后换掉报错的接口，有以下文件

pcdet/ops/pointnet2/pointnet2_batch/src/ball_query.cpp

pcdet/ops/pointnet2/pointnet2_batch/src/group_points.cpp

pcdet/ops/pointnet2/pointnet2_batch/src/interpolate.cpp

pcdet/ops/pointnet2/pointnet2_batch/src/sampling.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/ball_query_deform.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/ball_query.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/group_points.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/interpolate.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/sampling.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp

pcdet/ops/pointnet2/pointnet2_stack/src/vector_pool.cpp

pcdet/ops/votr_ops/src/build_attention_indices.cpp

pcdet/ops/votr_ops/src/build_mapping.cpp

pcdet/ops/votr_ops/src/group_features.cpp



将所有的

```c++
extern THCState *state;
```

注释掉

然后就

通了。。。。。。。



## 调试

根据git文档，输入

```sh
cd tools/PENet
python main.py --detpath /media/bopang/PBDATA/dataset/kitti/object/kitti/training
```

 报错缺库

```sh
ImportError: cannot import name 'bmat' from 'scipy.sparse.sputils' (/home/bopang/anaconda3/envs/pytorch3d/lib/python3.8/site-packages/scipy/sparse/sputils.py)
```



报错

```sh
ModuleNotFoundError: No module named 'tensorboardX'
```

重装升到了2.6.0解决



报错

```sh
ModuleNotFoundError: No module named 'prefetch_generator'
```



修改tools/cfgs/dataset_configs/kitti_dataset.yaml

line1中的路径到硬盘



pcdet/datasets/kitti/kitti_dataset.py

line444改称绝对路径，这里我这样修改

```python
if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        ROOT_DIR = (Path('/media/bopang/PBDATA/dataset/kitti/object/kitti').resolve()).resolve()
        # ROOT_DIR='/media/bopang/PBDATA/dataset/kitti/object/kitti'
        print(ROOT_DIR)
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            # data_path=ROOT_DIR / 'data' / 'kitti',
            # save_path=ROOT_DIR / 'data' / 'kitti'
            data_path=ROOT_DIR ,
            save_path=ROOT_DIR 
        )
```



此时报错

```
  File "/home/bopang/lidar_object_detection/TED/pcdet/datasets/kitti/kitti_dataset.py", line 142, in process_single_scene
    obj_list = self.get_label(sample_idx)
  File "/home/bopang/lidar_object_detection/TED/pcdet/datasets/kitti/kitti_dataset.py", line 74, in get_label
    assert label_file.exists()
AssertionErro
```

确实没有label_file.exists()



若没有修改，会报错

```
    assert img_file.exists()
AssertionError
```

而imgfile是存在的！



分析：

代码中索引的是

```python
label_file = self.root_split_path / 'label_2_semi' / ('%s.txt' % idx)
```

而实际上数据集中的是label_2

故修改pcdet/datasets/kitti/kitti_dataset.py的line73为

```python
label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
```



同理修改pcdet/datasets/kitti/kitti_dataset_mm.py的line470

与

哦，'label_2_semi'在这里写对了





现在可以进行train

此时train的操作为

```sh
cd tools
python train.py --cfg_file cfgs/models/kitti/TED-S.yaml
```



train的**parse_config**():可以调参

这里尝试--workers给到4

--batch_size给到3

epoch给到20

可以顶着显存跑

之后就是漫长的training







































