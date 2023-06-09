% 添加函数路径，需要deepMKL文件夹与本程序文件夹位于同路径下；
% 或根据开发环境需要修改。
% addpath('deepMKL');
clc;
traindata=csvread('PCAProjectedTrainData800.csv', 1, 0);
trainlabels=csvread('trainLabels.csv',1,0);
testdata=csvread('PCAProjectedTestData800.csv',1,0);
testlabels=csvread('testLabels.csv',1,0);
trainset = [traindata trainlabels];
testset = [testdata testlabels];

%fmst = computeFuzzynumber(trainset,0.001);
%[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcrossvalidationset([trainset fmst],5);
[tr_row,te_col]=size(traindata);
indices = crossvalind('Kfold',tr_row, 5);
% Indices = crossvalind('Kfold', N, K) %K折交叉验证
% 
% [Train, Test] = crossvalind('HoldOut', N, P) % 将原始数据随机分为两组,一组做为训练集,一组做为验证集
% 
% [Train, Test] = crossvalind('LeaveMOut', N, M) %留M法交叉验证，默认M为1，留一法交叉验证
% 
% [Train, Test] = crossvalind('Resubstitution', N, [P,Q])
% [...] = crossvalind(Method, Group, ...)
% [...] = crossvalind(Method, Group, ..., 'Classes', C)
% [...] = crossvalind(Method, Group, ..., 'Min', MinValue)
fuzzyMode = 'CMD';
C = 10;
MAX_ITER = 100;

%one layer
[model,net] = trainDFMKC(traindata,trainlabels,1,fuzzyMode,1E-5,MAX_ITER,C);
[pred,acc,cm,precision,recall,f1_score] = testDFMKC([traindata;testdata],testlabels,model,net);

%two layer
[model,net] = trainDFMKC(traindata,trainlabels,2,fuzzyMode,1E-5,MAX_ITER,C);
[pred,acc,cm,precision,recall,f1_score] = testDFMKC([traindata;testdata],testlabels,model,net);

%three layer
[model,net] = trainDFMKC(traindata,trainlabels,3,fuzzyMode,1E-5,MAX_ITER,C);
[pred,acc,cm,precision,recall,f1_score] = testDFMKC([traindata;testdata],testlabels,model,net);

% outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};
% 
% % 创建一个结构体数组，用于存储训练结果和测试结果
% results = struct('gauss_tr', {}, 'predict_gauss_tr', {}, 'predict_gauss_ts', {});
% 
% % 循环遍历所有训练集，进行训练和测试
% for n = 1:5
%     testindex = (indices==n);
%     trainindex = ~testindex;
%     testSingle = traindata(testindex,1:te_col);
%     fmstSingle2 = fmst(testindex,1);
%     testSingleLabel=trainlabels(testindex,1);
%     trainSingle = traindata(trainindex,1:te_col);
%     fmstSingle1 = fmst(trainindex,1);
%     trainSingleLabel=trainlabels(trainindex,1);
% 
%     eval(['[gauss_tr' num2str(n) ',boundary] = trainFSVC([trainSingle trainSingleLabel],fmstSingle1,C,''gauss'',gausskp);']);
%     eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,[trainSingle trainSingleLabel],[testSingle testSingleLabel],''gauss'',gausskp);']);
%     outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
%     outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
%     eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,[trainSingle trainSingleLabel],testset,''gauss'',gausskp);']);
%     outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
%     outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
% 
%     % % 训练FSVC模型
%     % [results(n).gauss_tr, boundary] = trainFSVC([trainSingle trainSingleLabel], fmstSingle1, C, 'gauss', gausskp);
%     % 
%     % % 测试FSVC模型在训练集上的表现
%     % [results(n).predict_gauss_tr, predicty_gauss_tr] = testFSVC(results(n).gauss_tr, boundary, [trainSingle trainSingleLabel], [testSingle testSingleLabel], 'gauss', gausskp);
%     % 
%     % % 测试FSVC模型在测试集上的表现
%     % [results(n).predict_gauss_ts, predicty_gauss_ts] = testFSVC(results(n).gauss_tr, boundary, [trainSingle trainSingleLabel], testset, 'gauss', gausskp);
%     % 
%     % % 将预测结果存储到输出表格中
%     % outputtable(2*n, 1) = {['predict_gauss_tr' num2str(n)]};
%     % outputtable(2*n, 2:19) = num2cell(results(n).predict_gauss_tr(:, 1:18));
%     % outputtable(2*n+1, 1) = {['predict_gauss_ts' num2str(n)]};
%     % outputtable(2*n+1, 2:19) = num2cell(results(n).predict_gauss_ts(:, 1:18));
% end
