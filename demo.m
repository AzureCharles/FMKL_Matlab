% Dorothea classification via SVC Matlab
X_train=csvread('PCAProjectedTrainData800.csv',1,0);
y_train=csvread('trainLabels.csv',1,0);
X_test=csvread('PCAProjectedTestData800.csv',1,0);
y_test=csvread('testLabels.csv',1,0);
C=100;
% load the data
mode = ('-c 100 -t 3 -q 1');
model = svmtrain(y_train, X_train, mode);
[y_pred,acc,proba] = svmpredict(y_test,X_test,model);

[cm,precision,recall,f1_score]=getF1Score(y_test,y_pred);



