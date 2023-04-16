# MKL
by Chat-gpt3.5

### 
## trainFSVC.m:
这段代码实现了深度多核学习（deepMKL）的训练过程，使用的是交替最小化算法。具体实现如下：

1. 首先读取输入的训练数据和标签，以及模型的参数设置，包括网络的层数、学习率、最大迭代次数和SVM的惩罚参数。

2. 初始化网络的权重参数betas，以及多核学习模型的核函数。

3. 进行交替优化，每次交替包括以下步骤：

   a. 使用当前的权重参数betas，计算多核学习模型的核函数，并使用LIBSVM库训练SVM分类器；

   b. 基于训练好的SVM分类器，计算损失函数的梯度，并根据网络的层数选择不同的梯度计算方式；

   c. 根据梯度和学习率更新网络的权重参数betas，并进行投影到可行域内；

   d. 判断停止条件，如果网络的权重参数没有发生明显变化，或者迭代次数达到了最大值，则停止迭代。

4. 返回训练好的SVM模型以及网络的参数，包括权重参数betas、核函数的参数sig、网络的层数和训练数据的大小。

## testFSVC.m:

这是一个 Matlab 函数，用于进行支持向量分类器（FSVC）的测试。输入参数包括 lambda、boundary、train、test、ker 和 para，输出包括 predictedY 和 stat。具体实现如下：

首先，从输入参数 train 中提取特征和标签，分别存储在变量 X 和 Y 中。然后从输入参数 test 中提取特征，存储在变量 Xt 中。接下来，通过 kernel 函数计算训练数据和测试数据之间的核矩阵 Kmatrix。最后，利用训练出的模型和核矩阵 Kmatrix 对测试数据进行分类，得到预测标签 predictedY。

最后，根据预测标签 predictedY 和实际标签 test 计算各种统计指标，例如准确率、精度、召回率、敏感度、特异度等，存储在 stat 中并返回。

注释中还有一段代码被注释掉了，看起来是用于计算 ks 和 gini 指标的，但是没有被调用。

## FSVC.m:

```%****************************FSVC models for classification*******************************%
% [trainset,testset] = partition(iris,41,41);
% fmst = computeFuzzynumber(trainset,0.001);
% cleanset = deleteoutliers(trainset,fmst,0.01);
% [tr1,tr2,tr3,tr4,fms1,fms2,fms3,fms4,vl1,vl2,vl3,vl4] = createcorssvalidationset(cleanset,4);

% minmax std
iid10ptrn1new = iid10ptrn1(:,[24 85 102 103 106 107 108 111 117]);
[trainset,testset] = partition(iid10ptrn1new,200,200);
iid10ptsn1new = iid10ptsn1(:,[24 85 102 103 106 107 108 111 117]);
testset = iid10ptsn1new;
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


iid10ptrn1new = iid10ptrn1(:,[24 79 81 87 105 106 117]);
[trainset,testset] = partition(iid10ptrn1new,200,200);
iid10ptsn1new = iid10ptsn1(:,[24 79 81 87 105 106 117]);
testset = iid10ptsn1new;
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


% no minmax std
[trainset,testset] = partition(iid10ptrn1(:,[73 80 117]),150,150);
testset = iid10ptsn1(:,[73 80 117]);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);

[trainset,testset] = partition(iid10ptrn2(:,[3 85 109 117]),150,150);
testset = iid10ptsn2(:,[3 85 109 117]);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);
```

这段代码包含了多个步骤，以下是每个步骤的简要描述：

1. 通过 partition 函数将数据集 iris 分为训练集和测试集，并计算训练集的模糊数值（fmst），然后使用 deleteoutliers 函数从训练集中删除异常值，最后使用 createcorssvalidationset 函数生成交叉验证集。
2. 从 iid10ptrn1 数据集中选择一些特定的列作为特征，然后使用 partition 函数将数据集分为训练集和测试集，并计算训练集的模糊数值（fmst），最后使用 createcorssvalidationset 函数生成交叉验证集。
3. 从 iid10ptrn1 数据集中选择另一组特定的列作为特征，然后使用 partition 函数将数据集分为训练集和测试集，并计算训练集的模糊数值（fmst），最后使用 createcorssvalidationset 函数生成交叉验证集。
4. 从 iid10ptrn1 和 iid10ptsn1 数据集中选择特定的列作为特征，然后使用 partition 函数将数据集分为训练集和测试集，并计算训练集的模糊数值（fmst），最后使用 createcorssvalidationset 函数生成交叉验证集。
5. 从 iid10ptrn2 和 iid10ptsn2 数据集中选择特定的列作为特征，然后使用 partition 函数将数据集分为训练集和测试集，并计算训练集的模糊数值（fmst），最后使用 createcorssvalidationset 函数生成交叉验证集。

每个步骤的具体实现可能需要查看函数的定义以及数据集的结构才能深入分析。

```trainset = aus14std_tr;
testset = aus14std_ts;

trainset = ger24std_tr;
testset = ger24std_ts;

trainset = usa66std_tr;
testset = usa66std_ts;

fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


%------------------------------linear kernel---------------------------%
C = 100;

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[linear_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20120126\outputFSVCforiid10pn120120522.xls',outputtable,'sheet1','A1');

%[ks gini] = createksandgini(score_linear_ts1);```
```

这段代码中，首先将三个数据集 `aus14std_tr`、`aus14std_ts`、`ger24std_tr`、`ger24std_ts`、`usa66std_tr` 和 `usa66std_ts` 分别赋值给 `trainset` 和 `testset`，然后使用 `computeFuzzynumber` 函数计算一个模糊数 `fmst`。接着，使用 `createcorssvalidationset` 函数将 `trainset` 和 `fmst` 拆分为十份，分别赋值给 `tr1` 到 `tr10`、`fms1` 到 `fms10` 和 `vl1` 到 `vl10`。

接下来的代码是使用线性核函数进行十次交叉验证训练和测试，并将结果存储在 `predict_linear_tr1` 到 `predict_linear_tr10` 和 `predict_linear_ts1` 到 `predict_linear_ts10` 中。每次训练和测试结束后，将相关结果记录到 `outputtable` 中，并将 `outputtable` 存储到 Excel 文件中。最后，该代码注释掉了计算 `ks` 和 `gini` 的部分，可能是因为这部分代码没有用处或者被移到了其他地方。
