%--------------------------------------------------------------------------------------------%
%------------------------------Call these functions of Models--------------------------------%
%--------------------------------Using sparse data format------------------------------------%
%-----------------------------------Designed by zhangzw--------------------------------------%
%-----------------------------------------2011.07.18-----------------------------------------%
%--------------------------------------------------------------------------------------------%

%****************************KFMCP models for classification*********************************%
load iris.mat;
[trainset,testset] = partition(iris,41,41);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.01);
[tr1,tr2,tr3,tr4,fms1,fms2,fms3,fms4,vl1,vl2,vl3,vl4] = createcorssvalidationset(cleanset,4);

%=====================================================================================================================================%
%*****************************austrilian****20110808************************************************%
load aus14std.mat;
[trainset,testset] = partition(aus14std,251,251);
[trainset,testset] = partition(aus14std,250,250);

[trainset,testset] = partition(aus14std,201,201);
[trainset,testset] = partition(aus14std,200,200);


%********************************china****20110808**************************************************%
load chn20std.mat;
[trainset,testset] = partition(chn20std,251,251);
[trainset,testset] = partition(chn20std,250,250);


%********************************german****20110808*************************************************%
load ger24std.mat;
[trainset,testset] = partition(ger24std,251,251); 
[trainset,testset] = partition(ger24std,250,250);

load ger30std.mat;
[trainset,testset] = partition(ger30std,201,201);


%********************************USA****20110808****************************************************%
load usa66std.mat;
[trainset,testset] = partition(usa66std,251,251);
[trainset,testset] = partition(usa66std,201,201);
[trainset,testset] = partition(usa66std,250,250);


%--------------------------------------------------------------------------------------------------%
trainset = aus14std_tr;
testset = aus14std_ts;

trainset = ger24std_tr;
testset = ger24std_ts;

trainset = usa66std_tr;
testset = usa66std_ts;

%------------------------------linear kernel-------------------------------%
C1 = 70;
C2 = 20;

fmst = computeFuzzynumber1(trainset,0.001,'linear',[]);
% fmst = computeFuzzynumber2(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset(cleanset,10);
% [tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[linear_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',C1,C2,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforAus1401.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel---------------------------%
C1 = 70;
C2 = 10;

polykp = [1,2];

fmst = computeFuzzynumber1(trainset,0.001,'poly',polykp);
% fmst = computeFuzzynumber2(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset(cleanset,10);
% [tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[poly_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',C1,C2,''poly'',polykp);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testKFMCP2(poly_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''poly'',polykp);']);
    outputtable(2*n,1) = {['predict_poly_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testKFMCP2(poly_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''poly'',polykp);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforAus1401.xls',outputtable,'sheet2','A1');

%------------------------------gauss kernel--------------------------------%
C1 = 10;
C2 = 20;

gausskp = 10;

fmst = computeFuzzynumber1(trainset,0.001,'gauss',gausskp);
% fmst = computeFuzzynumber2(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset(cleanset,10);
% [tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[gauss_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',C1,C2,''gauss'',gausskp);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testKFMCP2(gauss_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''gauss'',gausskp);']);
    outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testKFMCP2(gauss_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''gauss'',gausskp);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforAus1401.xls',outputtable,'sheet3','A23');

%------------------------------sigmoid kernel------------------------------%
C1 = 10;
C2 = 20;

sigmkp = [0.1,-0.01];


fmst = computeFuzzynumber1(trainset,0.001,'sigm',sigmkp);
% fmst = computeFuzzynumber2(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset(cleanset,10);
% [tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[sigm_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',C1,C2,''sigm'',sigmkp);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testKFMCP2(sigm_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''sigm'',sigmkp);']);
    outputtable(2*n,1) = {['predict_sigm_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testKFMCP2(sigm_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''sigm'',sigmkp);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforAus1401.xls',outputtable,'sheet4','A45');




%=====================================================================================================================================================%
%*******************************hp1bin6*****20110624**********************************************%
hp1bin61 = hp1bin6(:,[1,2,3,4,5,6,7,9,10,12,14,15]);

%delete noise
[trainset,testset] = partition(hp1bin61,201,201);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset(cleanset,5);

% [trainset,testset] = partition(hp1bin61,201,201);
% fmst = computeFuzzynumber(trainset,0.001);
% cleanset = deleteoutliers(trainset,fmst,0.001);
% [tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset(cleanset,10);

%hold noise
[trainset,testset] = partition(hp1bin61,200,200);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);


%********************************Call k-fold cross validation********for hp1bin6*******************************%
%------------------------------linear kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[linear_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp120110718.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[poly_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''poly'',[1,2]);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testKFMCP2(poly_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''poly'',[1,2]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testKFMCP2(poly_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''poly'',[1,2]);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp120110718.xls',outputtable,'sheet2','A1');

%------------------------------gauss kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[gauss_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''gauss'',0.01);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testKFMCP2(gauss_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''gauss'',0.01);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testKFMCP2(gauss_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''gauss'',0.01);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp120110718.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[sigm_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''sigm'',[0.001,-0.0001]);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testKFMCP2(sigm_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''sigm'',[0.001,-0.0001]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testKFMCP2(sigm_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''sigm'',[0.001,-0.0001]);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp120110718.xls',outputtable,'sheet4','A1');


%*******************************************************************************************************************%
%###################################################################################################################%
%*******************************************************************************************************************%

%*******************************hp2bin6*****20110624**********************************************%
hp2bin61 = hp2bin6(:,[1,2,3,4,5,7,9,10,12,14,15]);

%delete noise 
[trainset,testset] = partition(hp2bin61,71,71);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset(cleanset,5);

%hold noise
[trainset,testset] = partition(hp2bin61,70,70);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);

%****************************Call k-fold cross validation********for hp2bin6****************************************%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[linear_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[poly_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''poly'',[1,2]);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testKFMCP2(poly_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''poly'',[1,2]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testKFMCP2(poly_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''poly'',[1,2]);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet2','A1');

%------------------------------guass kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[gauss_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''gauss'',0.01);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testKFMCP2(gauss_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''gauss'',0.01);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testKFMCP2(gauss_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''gauss'',0.01);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[sigm_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr' num2str(n) ',fms' num2str(n) ',5,13,''sigm'',[0.001,-0.0001]);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testKFMCP2(sigm_tr' num2str(n) ',boundary, tr' num2str(n) ',tr' num2str(n) ',''sigm'',[0.001,-0.0001]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testKFMCP2(sigm_tr' num2str(n) ',boundary, tr' num2str(n) ',testset,''sigm'',[0.001,-0.0001]);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet4','A1');

%*******************************************************************************************************************%
%###################################################################################################################%
%*******************************************************************************************************************%

%###############################Parameter selection based on Grid search############################################%

%********************************Parameter selection for model*********hp1bin6*********************************************%
%-------------hp1bin6-----------------------%
hp1bin61 = hp1bin6(:,[1,2,3,4,5,6,7,9,10,12,14,15]);

[trainset,testset] = partition(hp1bin61,201,201);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:12);
fms = cleanset(:,13);

%-------------Breastcancer_STD-----------------------%
Breastcancer_STD = minmax(Breastcancer);
[trainset,testset] = partition(Breastcancer_STD,71,71);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:10);
fms = cleanset(:,11);

%-------------Diabetes_STD-----------------------%
Diabetes_STD = minmax(Diabetes);
[trainset,testset] = partition(Diabetes_STD,201,201);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:9);
fms = cleanset(:,10);

%-------------Heartdisease_STD-----------------------%
Heartdisease_STD = minmax(Heartdisease);
[trainset,testset] = partition(Heartdisease_STD,91,91);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:14);
fms = cleanset(:,15);

%-------------Hepatitis_STD---------------------------%
Hepatitis_STD = minmax(Hepatitis);
[trainset,testset] = partition(Hepatitis_STD,25,25);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.000001);
tr = cleanset(:,1:20);
fms = cleanset(:,21);

%-------------Liverdisorder_STD-----------------------%
Liverdisorder_STD = minmax(Liverdisorder);
[trainset,testset] = partition(Liverdisorder_STD,101,101);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:7);
fms = cleanset(:,8);

%-------------Lungcancer_STD-----------------------%
Lungcancer_STD = minmax(Lungcancer);
[trainset,testset] = partition(Lungcancer,6,6);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.000001);
tr = cleanset(:,1:57);
fms = cleanset(:,58);

Lungcancer_STD = minmax(Lungcancer);
trainset = Lungcancer_STD;
testset = Lungcancer_STD;
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.000001);
tr = cleanset(:,1:57);
fms = cleanset(:,58);


%-------------Spectfheartdisease_STD-----------------------%
Spectfheartdisease_STD = minmax(Spectfheartdisease);
[trainset,testset] = partition(Spectfheartdisease_STD,41,41);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:45);
fms = cleanset(:,46);


%-------------Spectheartdisease_STD-----------------------%
Spectheartdisease_STD = minmax(Spectheartdisease);
[trainset,testset] = partition(Spectheartdisease_STD,41,41);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:23);
fms = cleanset(:,24);

%-------------Wbreastcancer_STD-----------------------%
Wbreastcancer_STD = minmax(Wbreastcancer);
[trainset,testset] = partition(Wbreastcancer_STD,201,201);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:10);
fms = cleanset(:,11);

%----------------------------------------------------%
C1 = [16,32,64,128,256,512,1024];
C2 = [1,10,20,50,100,200,500,1000];

%------------------------------linear kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:8
    for m=1:1:8
        [linear_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'linear',[]);
        [predicty_linear_ts, predict_linear_ts] = testKFMCP2(linear_tr,boundary, tr,testset,'linear',[]); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_linear_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPAus1420110819.xls',outputtable,'sheet1','A52');

%------------------------------polynomial kernel---------------------------%
epsilon = 1e+6;  
polykp = [0,2];
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [poly_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'poly',polykp);
        [predicty_poly_ts, predict_poly_ts] = testKFMCP2(poly_tr,boundary, tr,testset,'poly',polykp); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_poly_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPAus1420110819.xls',outputtable,'sheet2','A52');

%------------------------------guass kernel---------------------------%
epsilon = 1e+6;  
gausskp = 50;
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [gauss_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'gauss',gausskp);
        [predicty_gauss_ts, predict_gauss_ts] = testKFMCP2(gauss_tr,boundary, tr,testset,'gauss',gausskp); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_gauss_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPAus1420110819.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
epsilon = 1e+6;  
sigmkp = [0.01,-0.001];
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [sigm_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'sigm',sigmkp);
        [predicty_sigm_ts, predict_sigm_ts] = testKFMCP2(sigm_tr,boundary, tr,testset,'sigm',sigmkp); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_sigm_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPAus1420110819.xls',outputtable,'sheet4','A1');


%********************************Parameter selection for model*********hp2bin6*********************************************%
hp2bin61 = hp2bin6(:,[1,2,3,4,5,7,9,10,12,14,15]);

[trainset,testset] = partition(hp2bin61,71,71);
fmst = computeFuzzynumber(trainset,0.001);
cleanset = deleteoutliers(trainset,fmst,0.001);
tr = cleanset(:,1:11);
fms = cleanset(:,12);

C1 = [16,32,64,128,256,512,1024];
C2 = [10,20,30,50,70,90,100,200,300,500,700,900,1000];
C3 = [10,20,50,100,200,500,1000];

%------------------------------linear kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [linear_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C1(n),C1(m),'linear',[]);
        [predicty_linear_ts, predict_linear_ts] = testKFMCP2(linear_tr,boundary, tr,testset,C1(n),C1(m),'linear',[]); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_linear_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [poly_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C1(n),C1(m),'poly',[1,2]);
        [predicty_poly_ts, predict_poly_ts] = testKFMCP2(poly_tr,boundary, tr,testset,C1(n),C1(m),'poly',[1,2]); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_poly_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet2','A1');

%------------------------------guass kernel---------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [gauss_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C1(n),C1(m),'gauss',0.01);
        [predicty_gauss_ts, predict_gauss_ts] = testKFMCP2(gauss_tr,boundary, tr,testset,C1(n),C1(m),'gauss',0.01); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_gauss_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel------------------------------%
epsilon = 1e+6;  
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:7
    for m=1:1:7
        [sigm_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C1(n),C1(m),'sigm',[0.01,-0.001]);
        [predicty_sigm_ts, predict_sigm_ts] = testKFMCP2(sigm_tr,boundary, tr,testset,C1(n),C1(m),'sigm',[0.01,-0.001]); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_sigm_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforhp220110718.xls',outputtable,'sheet4','A1');


%***********************************************************************************************************************%
%-----------------------------------------designed by zhangzw on Aug. 09, 2011------------------------------------------%
%***********************************************************************************************************************%
%-------------------------Parameter selection for model----------------------credit scoring-----------------------------%
C1 = [2,4,8,16,32,64,128,256,512,1024];
C2 = [10,20,50,100,200,500,1000,2000,5000,10000];
C3 = [10,20,30,50,70,90,100,200,300,500,700,900];
C4 = [10,20,30,40,50,60,70,80,90,100];

%------------------------------linear kernel---------------------------%
%-------------AUS_STD-----------------------%
% delete noise
[trainset,testset] = partition(aus14std,200,200);
fmst = computeFuzzynumber1(trainset,0.001,'linear',[]);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

% hold noise
% [trainset,testset] = partition(aus14std,200,200);
% fmst = computeFuzzynumber1(trainset,0.001,'linear',[]);
% tr = trainset;
% fms = fmst;

%-------------CHN_STD-----------------------%
[trainset,testset] = partition(chn20std,201,201);
fmst = computeFuzzynumber1(trainset,0.001,'linear',[]);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------GER_STD-----------------------%
[trainset,testset] = partition(ger24std,201,201);
fmst = computeFuzzynumber1(trainset,0.001,'linear',[]);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------USA_STD-----------------------%
[trainset,testset] = partition(usa66std,201,201);
fmst = computeFuzzynumber1(trainset,0.001,'linear',[]);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%---------------SP for linear kernel---------------------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
     eval(['[linear_tr' num2str(n) ', boundary, alpha_tr' num2str(n) ', beta_tr' num2str(n) '] = trainKFMCP2(tr,fms,C1(n),C1(10-n+1),''linear'',[]);']);
     eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testKFMCP2(linear_tr' num2str(n) ',boundary, tr,testset,''linear'',[]);']);
     outputtable((n+1),1) = {n};
     outputtable((n+1),2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforSPAus1401.xls',outputtable,'sheet1','A1');


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:8
    for m=1:1:8
        [linear_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'linear',[]);
        [predicty_linear_ts, predict_linear_ts] = testKFMCP2(linear_tr,boundary, tr,testset,'linear',[]); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_linear_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforSPAus1401.xls',outputtable,'sheet1','A1');


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:12
    for m=1:1:12
        [linear_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C3(n),C3(m),'linear',[]);
        [predicty_linear_ts, predict_linear_ts] = testKFMCP2(linear_tr,boundary, tr,testset,'linear',[]); 
        outputtable((m+1)+(n-1)*12,1) = {m+(n-1)*12};
        outputtable((m+1)+(n-1)*12,2:19) = num2cell(predict_linear_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPUsa6620110824.xls',outputtable,'sheet1','A1');



%***********************************************************************************************************************%
%------------------------------polynomial kernel---------------------------%
polykp = [0,2];

%-------------AUS_STD-----------------------%
[trainset,testset] = partition(aus14std,200,200);
fmst = computeFuzzynumber1(trainset,0.001,'poly',polykp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------CHN_STD-----------------------%
[trainset,testset] = partition(chn20std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'poly',polykp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------GER_STD-----------------------%
[trainset,testset] = partition(ger24std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'poly',polykp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------USA_STD-----------------------%
[trainset,testset] = partition(usa66std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'poly',polykp);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%---------------SP for polynomial kernel-----------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:8
    for m=1:1:8
        [poly_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'poly',polykp);
        [predicty_poly_ts, predict_poly_ts] = testKFMCP2(poly_tr,boundary, tr,testset,'poly',polykp); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_poly_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforSPAus14.xls',outputtable,'sheet2','A1');


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:12
    for m=1:1:12
        [poly_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C3(n),C3(m),'poly',polykp);
        [predicty_poly_ts, predict_poly_ts] = testKFMCP2(poly_tr,boundary, tr,testset,'poly',polykp); 
        outputtable((m+1)+(n-1)*12,1) = {m+(n-1)*12};
        outputtable((m+1)+(n-1)*12,2:19) = num2cell(predict_poly_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPUsa6620110824.xls',outputtable,'sheet2','A147');

%***********************************************************************************************************************%
%------------------------------guass kernel---------------------------%
gausskp = 2;

%-------------AUS_STD-----------------------%
[trainset,testset] = partition(aus14std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'gauss',gausskp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------CHN_STD-----------------------%
[trainset,testset] = partition(chn20std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'gauss',gausskp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------GER_STD-----------------------%
[trainset,testset] = partition(ger24std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'gauss',gausskp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------USA_STD-----------------------%
[trainset,testset] = partition(usa66std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'gauss',gausskp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%---------------SP for guass kernel-----------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:8
    for m=1:1:8
        [gauss_tr, boundary, , beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'gauss',gausskp);
        [predicty_gauss_ts, predict_gauss_ts] = testKFMCP2(gauss_tr,boundary, tr,testset,'gauss',gausskp); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_gauss_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforSPAus14.xls',outputtable,'sheet3','A1');


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:12
    for m=1:1:12
        [gauss_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C3(n),C3(m),'gauss',gausskp);
        [predicty_gauss_ts, predict_gauss_ts] = testKFMCP2(gauss_tr,boundary, tr,testset,'gauss',gausskp); 
        outputtable((m+1)+(n-1)*12,1) = {m+(n-1)*12};
        outputtable((m+1)+(n-1)*12,2:19) = num2cell(predict_gauss_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPUsa6620110824.xls',outputtable,'sheet3','A147');

%***********************************************************************************************************************%
%------------------------------sigmoid kernel---------------------------%
sigmkp = [0.01,-0.001];

%-------------AUS_STD-----------------------%
[trainset,testset] = partition(aus14std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'sigm',sigmkp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------CHN_STD-----------------------%
[trainset,testset] = partition(chn20std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'sigm',sigmkp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------GER_STD-----------------------%
[trainset,testset] = partition(ger24std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'sigm',sigmkp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%-------------USA_STD-----------------------%
[trainset,testset] = partition(usa66std,201,201);
fmst = computeFuzzynumber(trainset,0.001,'sigm',sigmkp);
cleanset = deleteoutliers(trainset,fmst,0.001);
ncol = size(cleanset,2);
tr = cleanset(:,1:ncol-1);
fms = cleanset(:,ncol);

%---------------SP for sigm kernel-----------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:8
    for m=1:1:8
        [sigm_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C2(n),C2(m),'sigm',sigmkp);
        [predicty_sigm_ts, predict_sigm_ts] = testKFMCP2(sigm_tr,boundary, tr,testset,'sigm',sigmkp); 
        outputtable((m+1)+(n-1)*7,1) = {m+(n-1)*7};
        outputtable((m+1)+(n-1)*7,2:19) = num2cell(predict_sigm_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\results20120123\outputKFMCPforSPAus14.xls',outputtable,'sheet4','A1');


outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:12
    for m=1:1:12
        [sigm_tr, boundary, alpha_tr, beta_tr] = trainKFMCP2(tr,fms,C3(n),C3(m),'sigm',sigmkp);
        [predicty_sigm_ts, predict_sigm_ts] = testKFMCP2(sigm_tr,boundary, tr,testset,'sigm',sigmkp); 
        outputtable((m+1)+(n-1)*12,1) = {m+(n-1)*12};
        outputtable((m+1)+(n-1)*12,2:19) = num2cell(predict_sigm_ts(:,1:18));
    end
end

xlswrite('D:\KFMCPwithMatlab2\outputKFMCPforSPUsa6620110824.xls',outputtable,'sheet4','A147');


%*******************************************************************************************************************%
%###################################################################################################################%
%*******************************************************************************************************************%
%****************************************************************************************************************************************%


