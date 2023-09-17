clc;clear;
load ('ionosphere.mat');

%%
X1=mapminmax(X1',0,1);
X2=mapminmax(X2',0,1);

data=X1';
data2=X2';
y = y;
[M,N]=size(data);
%
a1 =0;
a2 = 0;
a3 = 0;
C = 10;
c = 0; 
%%
FunPara.C=C;
FunPara.c1=C;
FunPara.c2=C;
FunPara.c3=0.01;
FunPara.a1=10^(a1);
FunPara.a2=10^(a2);
FunPara.a3=10^(a3);
FunPara.kerfPara.type = 'rbf';
FunPara.kerfPara.pars = 10^c;
tic
indices=crossvalind('Kfold',M,5);              
for k=1:5
    %
    test = (indices == k);
    train = ~test;
    train_data=data(train,:);
    train_data2=data2(train,:);
    train_Y=y(train,:);
    test_data=data(test,:);
    test_data2=data2(test,:);
    test_Y=y(test,:);
    %
    DataTrain.Xa = train_data;
    DataTrain.Xb = train_data2;
    DataTrain.Y = train_Y;
    Test.Xa = test_data;
    Test.Xb = test_data2;
    Y = test_Y;
    [Predict_Y1,Predict_Y2,Predict_Y] = MVLDM(Test,DataTrain,FunPara);
    Accuracy1(k) = sum(Y == Predict_Y1)/size(Y,1);
    Accuracy2(k) = sum(Y == Predict_Y2)/size(Y,1);
    Accuracy(k) = sum(Y == Predict_Y)/size(Y,1);
end
fprintf('---------------------------------------------------------------------------------------------- %.4f\n',mean(Accuracy));
toc

