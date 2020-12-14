## 任务描述

印花布匹瑕疵分类，有如下任务（PPT上说一个任务是三分类，然后其他类别可自行分析进行合并分类）：

1. 取1、2、14共三类数据，按3类作分类
2. 取1、2、4、14、20共五类数据，按5类作分类
3. 取所有数据，按15类作分类

## 数据描述

数据集内印花布匹瑕疵被划分为多类，瑕疵以成对图片（正确的图样以及问题图样）给出，并注明了瑕疵的参考位置及类别。上述瑕疵图、模版图及瑕疵参考位置均可作为已知信息用于瑕疵类型的判别。

#### 布匹瑕疵类别

1. 逃花
2. 塞网
3. 破洞
4. 缝头
5. 水渍
6. 脏污
7. 白条

14. 未对齐

16. 伪色

17. 前后色差

20. 模板取错
21. 漏浆
22. 脱浆
23.  色纱（异纤)
24. 飞絮（线头）

#### 数据集的文件目录

* fabric_data_new
  * label_json
    * 1594408408928_dev001
    * 1594418800470_dev001
    * ...
  * temp
    * 1594408408928_dev001
    * 1594418800470_dev001
    * ...
  * trgt
    * 1594408408928_dev001
    * 1594418800470_dev001
    * ...

**fabric_data_new/label_json/**

标记文件

* 示例

  ```json
  {
  	"flaw_type": 4,   # 瑕疵类别
  	"bbox": {		   # bounding box，即瑕疵的位置
  		"x0": 125,
  		"x1": 188,
  		"y0": 171,
  		"y1": 230
  	}
  }
  ```

**fabric_data_new/temp/**

模板图，可作为参考的正确图样

**fabric_data_new/trgt/**

瑕疵图，含缺陷的布匹图片

**说明**

* 每个文件夹下包含各批次的图像，每一批次的印花图案相同，不同批次的印花图案可能相同
* 标记文件、模板图和瑕疵图的文件名称一一对应

## 数据预处理

#### load_data.py

* load_data(data_path)

  功能：

  * 读取数据集中的所有图片和标注，返回一个```datas```数组，数组每一个元素的格式为，```{'img1': img1, 'img2': img2, 'info': info}```。img1为模板图；img2为瑕疵图；info为标注，每一项为一个字典，包括瑕疵类别和瑕疵位置，格式为：{"flaw_class": \<int\>, "bbox": [\<int\>, \<int\>, \<int\>, \<int\>]}

  * 输出一个label.txt文件，记录各类别样本数量和各样本的一些信息，如"bbox"的最长边

    ```txt
    各类别样本数量
     1(逃花): 143
     2(塞网): 254
     3(破洞): 10
     4(缝头): 319
     5(水渍): 9
     6(脏污): 47
     7(白条): 8
    14(未对齐): 562
    16(伪色): 7
    17(前后色差): 25
    20(模板取错): 197
    21(漏浆): 7
    22(脱浆): 16
    23(色纱): 7
    24(飞絮): 23
    
    
    各样本信息
    maxl: 2048 flaw_type: 4 location: D://fabric_data_new/temp\1594451485172_dev001\938142228165_337_1_0.jpg x0: 0 x1: 2048 y0: 119 y1: 281
    maxl: 2048 flaw_type: 4 location: D://fabric_data_new/temp\1594698846538_dev001\22970991116964_377_1_0.jpg x0: 0 x1: 2048 y0: 111 y1: 290
    ......
    ```

    各类别样本的数量有较大的差异，需要做数据增强

#### process.py

* read_img(img)

  功能：将图片转换成numpy数组

* cut(img, bbox)

  功能：根据标注边框进行裁剪,返回裁剪后的局部图片

* diff(img1, img2)

  功能：返回两张图片的numpy数组差分后的numpy数组

* process(img1, img2, bbox)

  功能：对一对样本图片进行裁剪

* mkdir(root_path, flaw_type)

  功能：删除已经存在的数据目录，新建空的数据目录，```flaw_type```是瑕疵类别集合

* catagory(flaw_type)

  功能：把原始数据集的每对样本图片（模板图和瑕疵图）用```process```处理，然后把差分图按照瑕疵类别分类存放。对于瑕疵类别 x，把该类的差分图存进```args.catagory_cut_raw_data_path/typex```目录下

* split_data(flaw_type, file_path)

  功能：在```catagory```原始数据集后，划分训练集和测试集，```file_path```是划分后的训练集(```train.txt```)和测试集(```test.txt```)的存放路径（这里是按照文件名划分数据集的）。存储的内容是```list```，```list```每个元素的格式为：```[flaw_type, path]```。```flaw_type```是瑕疵类别，```path```是文件名。根据瑕疵类别和文件名就能确定一个差分图文件。例如```[1, pic1.jpg]```表示差分图文件路径```args.catagory.../type1/pic1.jpg```

* pre_aug(flaw_type, file_path)

  功能：做增强数据集前的准备工作，创建目录，把训练集数据分类存储到```args.augmentated_data_path/```目录下

* aug_collection(flaw_type, flaw_count)

  功能：对每类瑕疵的训练集调用```augmentation.py```文件中的```transform_image```进行数据增强，把每类瑕疵的训练集样本数增加到```pic_number```。程序运行完，对于瑕疵类别x，```args.aug.../typex/```目录下有```pic_number```个图片

* conv2numpy(flaw_type, file_path)

  功能：从```args.augmentated_data_path/```读出增强后的训练集数据，根据```test.txt```从```args.catagory_cut_raw_data_path/```读出测试集数据，把训练集和测试集的数据图片```resize```后转换成numpy数组存储起来。最后，对于任务```x```(1,2,3)，```file_path/taskx/```路径下存放有训练集数据```x_train.npy```，```y_train.npy```，测试集数据```x_test.npy```，```y_test.npy```。此时瑕疵类别已经转换成0, 1, ..., 14，而不是1, 2, ..., 24了

#### augmentation.py

* random_flip_left_right(image)

  功能：随机左右翻转图片

* random_flip_up_down(image)

  功能：随机上下翻转图片

* random_contrast(image, minval=0.6, maxval=1.4)

  功能：随机改变图片的对比度

* random_brightness(image, minval=0., maxval=.2)

  功能：随机改变图片的亮度

* random_saturation(image, minval=0.4, maxval=2.)

  功能：随机改变图片的饱和度

* random_hue(image, minval=-0.04, maxval=0.08)

  功能：随机改变图片的色调

* tf_rotate(input_image, min_angle = -np.pi/2, max_angle = np.pi/2)

  功能：把图片随机旋转一个角度

* transform_image(image)

  功能：把一个图片使用上面的所有方法（外加一个随机旋转90度）进行变换

* 说明：对于样本数量较多的类别[1, 2, 4, 14, 20]，没有使用上述所有的数据增强方法

经过数据预处理，在```file_path/taskx/```路径下存放着任务```x```所需的训练集和测试集的```.npy```数据文件

## 算法介绍

### 简单cnn

#### test1.py



### 逻辑回归

#### test2.py



## 实验结果

|               |                           简单cnn                            |   逻辑回归   |
| :-----------: | :----------------------------------------------------------: | :----------: |
| task1(3分类)  |       average:0.79 class0:0.65 class1:0.84 class2:0.79       | average:0.56 |
| task2(5分类)  | average:0.74 class0:0.65 class1:0.83 class2:0.86 class3:0.68 class4:0.68 | average:0.48 |
| task3(15分类) |                           结果太差                           |   结果太差   |

