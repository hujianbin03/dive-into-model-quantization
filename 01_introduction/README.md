### 1. 基础
#### 1.1 什么是模型量化
**定义**：量化Quantization，是指将高精度浮点数(如float32)表示为低精度整数(如int8)的过程，从而提高神经网络的效率和性能。  
常见的模型量化技术包括**权重量化**和**激活量化**  
* **权重量化**：是将浮点参数转换为整数参数，常用的量化方法包括对称量化和非对称量化
  * 对称量化：是将权重范围均匀的分配在正负两个方向，将浮点数据映射到一个整数范围中。
  * 非对称量化：是将权重范围分配在一个方向，即只使用非负整数表示。
* **激活量化**：是将输入和输出数据转换为低比特宽度的数据类型，同样可以采用对称量化和非对称量化两种方法。  
**注意**：模型量化会对模型的精度和准确度产生一定的影响，因为量化后的模型可能无法完全保留原始模型中的所有信息和特征。因此，在
进行模型量化时需要进行适当的权衡和优化。

我们平时训练处的模型如YOLOv5、ResNet50正常到处默认都时FP32的精度

#### 1.2 为什么要学习模型量化
训练好的模型权重一般来说时FP32，即单精度浮点型，在深度学习训练和推理的过程中，最常见的精度就是FP32.  
* **FP32**是单精度浮点数，采用32位二进制表示，其中1位为符号位，8位为指数位，23位为尾数位。  
* **FP16**是半精度浮点数，采用16位二进制表示，其中1位为符号位，5位为指数位，10位为尾数位。  
* **INT8**是8位整数，采用8位二进制表示，其中1位位符号位，7位为数值位。
对于浮点数来说，指数位表示**该精度可达到的动态范围**，而尾数位表示精度。  

从FP32=>FP16是一种量化，只不过因为FP32=>FP16几乎是无损的，不需要**calibrator**去校正、更不需要
**retrain**。 并且FP16的精度下降对于大部分任务影响不是很大，甚至有些任务会提升。NVIDIA对于FP16有
专门的Tensor Cores可以进行矩阵运算，**相比于FP32来说吞吐量直接提升一倍，提速效果明显**。

实际来说，**量化就是将我们训练好的模型，不论是权重，还是计算op，都转换为低精度去计算**。实际中我们谈论
的量化更多的是**INT8量化**。

在深度学习中，量化有以下优势：
* 减少内存占用，模型容量变小，如FP32权重变成INT8，大小直接缩小了4倍数。
* 加速计算，实际卷积计算的op是INT8类型，在特定硬件下可以利用INT8的指令集去实现高吞吐，不论是GPU、INTEL、
ARM等平台都有**INT8的指令集优化**。
* 减少功耗和延迟，有利用嵌入式侧设备的应用。
* 量化是模型部署中的**一种重要的优化方法**，可以在**部分精度损失**的前提下，大幅度提高神经网络的**效率和性能**。

#### 1.3 如何学习模型量化
量化的两个重要过程，**量化(Quantize)**和**反量化(Dequantize)**
* 量化：就是将浮点数量化为整型数(FP32=>INT8)
* 反量化：将整型数转换为浮点数(INT8=>FP32)

将一个浮点数转换为整型数 (5.214)
1. 计算线性映射的缩放值Scale  
Scale = 5.214 / 127 = 0.0410551
2. 量化操作  
5.214 / 0.0412126 ==> Round(127.000056) ==> 127 ==> 01111111
3. 反量化操作   
01111111 ==> 127 X 0.0410551 ==> 5.213997  

将一个浮点数组转换为整数型 [-0.62, -0.52, 1.62]
1. 计算数组共同Scale  
``` 
Scale = (float_max - float_min) / (quant_max - quant_min)  
      = (1.62 - (-0.61)) / (127 - (-128))  
      = 0.0087109
```
2. 量化  
```
-0.61 / 0.0087109 = -70.0272072
-0.52 / 0.0087109 = -59.6953242	
1.62 / 0.0087109 = 185.9738947
==> [-70,-59,185] 取整
```
3. 截断  
```
[-70,-59,185] ==> [-70,-59,127]
```
4. 反量化  
```
[-0.609763,-0.5139431,1.1062843]
```
可以看到截断的数值最后反量化与原数值相差较大(1.62与1.1062843)

如何解决这个问题：1、最大绝对值对称法 2、偏移





















