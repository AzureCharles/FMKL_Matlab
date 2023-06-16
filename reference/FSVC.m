%--------------------------------------------------------------------------------------------%
%------------------------------Call these functions of Models--------------------------------%
%--------------------------------Using sparse data format------------------------------------%
%-----------------------------------Designed by zhangzw--------------------------------------%
%-----------------------------------------2011.05.19-----------------------------------------%
%--------------------------------------------------------------------------------------------%

%****************************FSVC models for classification*******************************%
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




%************************************************************************************************************************%
%*****************************austrilian****20110608************************************************%
[trainset,testset] = partition(aus14std,200,200);
[trainset,testset] = partition(aus14std,250,250);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


%*****************************chn****20110808******************************************************%
[trainset,testset] = partition(chn20std,200,200);
[trainset,testset] = partition(chn20std,250,250);
[trainset,testset] = partition(chn20std12,250,250);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


%********************************german****20110617************************************************%
[trainset,testset] = partition(ger24std,200,200);
[trainset,testset] = partition(ger24std,250,250);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


%*****************************usa****20110608******************************************************%
[trainset,testset] = partition(usa66std,150,150);
[trainset,testset] = partition(usa66std,250,250);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9,tr10,fms1,fms2,fms3,fms4,fms5,fms6,fms7,fms8,fms9,fms10,vl1,vl2,vl3,vl4,vl5,vl6,vl7,vl8,vl9,vl10] = createcorssvalidationset([trainset fmst],10);


%--------------------------------------------------------------------------------------------------%
trainset = aus14std_tr;
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

%[ks gini] = createksandgini(score_linear_ts1);

%------------------------------polynomial kernel-------------------------%
C = 100;
polykp = [0,2];

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[poly_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''poly'',polykp);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''poly'',polykp);']);
    outputtable(2*n,1) = {['predict_poly_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''poly'',polykp);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20120126\outputFSVCforiid10pn120120522.xls',outputtable,'sheet2','A1');

%------------------------------gauss kernel------------------------------%
C = 100;
gausskp = 10;

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[gauss_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''gauss'',gausskp);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''gauss'',gausskp);']);
    outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''gauss'',gausskp);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20120126\outputFSVCforiid10pn120120522.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
C = 100;
sigmkp = [1,-0.1];

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:10
    eval(['[sigm_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''sigm'',sigmkp);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''sigm'',sigmkp);']);
    outputtable(2*n,1) = {['predict_sigm_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''sigm'',sigmkp);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20120126\outputFSVCforiid10pn120120522.xls',outputtable,'sheet4','A1');

%*****************************************************************************************************************%
%-----------------------------------------------------------------------------------------------------------------%
%*************************************20140111******ppi***********************************************************%
load('pssmprobtrmmred');
load('pssmprobtsmmred');
testset = pssmprobtsmmred;
[trainset,valset] = partition(pssmprobtrmmred,1000,1000);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);

load('pssmprobtrstd');
load('pssmprobtsstd');
testset = pssmprobtsstd;
[trainset,valset] = partition(pssmprobtrstd,1000,1000);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);

%*******************************AID644red*********************************************************************%
load('AID644redtrain');
load('AID644redtest');

[AID644redtrain1,AID644redtest1] = minmax2dataset(AID644redtrain,AID644redtest);
testset = AID644redtest1;
[trainset,valset] = partition(AID644redtrain1,111,54);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);



%------------------------------linear kernel---------------------------%
C = 10;

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[linear_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140111\outputFSVCforAID644red20140112.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel-------------------------%
C = 10;
polykp = [1,2];

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[poly_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''poly'',polykp);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''poly'',polykp);']);
    outputtable(2*n,1) = {['predict_poly_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''poly'',polykp);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140111\outputFSVCforAID644red20140112.xls',outputtable,'sheet2','A1');

%------------------------------gauss kernel------------------------------%
C = 1000;
gausskp = 10;

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[gauss_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''gauss'',gausskp);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''gauss'',gausskp);']);
    outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''gauss'',gausskp);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140111\outputFSVCforAID644red20140112.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
C = 1;
sigmkp = [0.1,-0.01];

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[sigm_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''sigm'',sigmkp);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''sigm'',sigmkp);']);
    outputtable(2*n,1) = {['predict_sigm_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''sigm'',sigmkp);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140111\outputFSVCforAID644red20140112.xls',outputtable,'sheet4','A1');

%*****************************************************************************************************************%
%-----------------------------------------------------------------------------------------------------------------%
%*************************************20140212******AID***********************************************************%
%--------------------AID644red---------------------%
load('AID644redtrain');
load('AID644redtest');

[AID644redtrmm,AID644redtsmm] = minmax2dataset(AID644redtrain,AID644redtest);
testset = AID644redtsmm;
[trainset,valset] = partition(AID644redtrmm,111,54);

%-------------------AID362red----------------------%
load('AID362redtrain');
load('AID362redtest');

[AID362redtrmm,AID362redtsmm] = minmax2dataset(AID362redtrain,AID362redtest);
testset = AID362redtsmm;
[trainset,valset] = partition(AID362redtrmm,3375,48);
%[trainset,valset] = partition(AID362redtrmm,2363,34);
%[trainset,valset] = partition(AID362redtrmm,1688,24);

%-------------------AID373red----------------------%
load('AID373redtrain');
load('AID373redtest');

[AID373redtrmm,AID373redtsmm] = minmax2dataset(AID373redtrain,AID373redtest);
testset = AID373redtsmm;
[trainset,valset] = partition(AID373redtrmm,250,50);

%-------------------AID439red----------------------%
load('AID439redtrain');
load('AID439redtest');

[AID439redtrmm,AID439redtsmm] = minmax2dataset(AID439redtrain,AID439redtest);
testset = AID439redtsmm;
[trainset,valset] = partition(AID439redtrmm,45,11);

%-------------------AID456red----------------------%
load('AID456redtrain');
load('AID456redtest');

[AID456redtrmm,AID456redtsmm] = minmax2dataset(AID456redtrain,AID456redtest);
testset = AID456redtsmm;
[trainset,valset] = partition(AID456redtrmm,110,22);

%-------------------AID604red----------------------%
load('AID604redtrain');
load('AID604redtest');

[AID604redtrmm,AID604redtsmm] = minmax2dataset(AID604redtrain,AID604redtest);
testset = AID604redtsmm;
[trainset,valset] = partition(AID604redtrmm,350,150);

%-------------------AID687red----------------------%
load('AID687redtrain');
load('AID687redtest');

[AID687redtrmm,AID687redtsmm] = minmax2dataset(AID687redtrain,AID687redtest);
testset = AID687redtsmm;
[trainset,valset] = partition(AID687redtrmm,250,76);

%-------------------AID688red----------------------%
load('AID688redtrain');
load('AID688redtest');

[AID688redtrmm,AID688redtsmm] = minmax2dataset(AID688redtrain,AID688redtest);
testset = AID688redtsmm;
[trainset,valset] = partition(AID688redtrmm,350,150);

%-------------------AID721red----------------------%
load('AID721redtrain');
load('AID721redtest');

[AID721redtrmm,AID721redtsmm] = minmax2dataset(AID721redtrain,AID721redtest);
testset = AID721redtsmm;
[trainset,valset] = partition(AID721redtrmm,59,17);

%-------------------AID746red----------------------%
load('AID746redtrain');
load('AID746redtest');

[AID746redtrmm,AID746redtsmm] = minmax2dataset(AID746redtrain,AID746redtest);
testset = AID746redtsmm;
[trainset,valset] = partition(AID746redtrmm,300,200);

%-------------------AID1284red----------------------%
load('AID1284redtrain');
load('AID1284redtest');

[AID1284redtrmm,AID1284redtsmm] = minmax2dataset(AID1284redtrain,AID1284redtest);
testset = AID1284redtsmm;
[trainset,valset] = partition(AID1284redtrmm,244,46);

%-------------------AID1608red----------------------%
load('AID1608redtrain');
load('AID1608redtest');

[AID1608redtrmm,AID1608redtsmm] = minmax2dataset(AID1608redtrain,AID1608redtest);
testset = AID1608redtsmm;
[trainset,valset] = partition(AID1608redtrmm,245,55);


%*******************************************************************%
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);



%------------------------------linear kernel---------------------------%
C = 10;

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[linear_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140212\outputFSVCforAID362red20140212-new.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel-------------------------%
C = 1000;
polykp = [1,2];

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[poly_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''poly'',polykp);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''poly'',polykp);']);
    outputtable(2*n,1) = {['predict_poly_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''poly'',polykp);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140212\outputFSVCforAID644red20140212.xls',outputtable,'sheet2','A1');

%------------------------------gauss kernel------------------------------%
C = 0.001;
gausskp = 5;

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[gauss_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''gauss'',gausskp);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''gauss'',gausskp);']);
    outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''gauss'',gausskp);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140212\outputFSVCforAID362red20140212-new.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
C = 1000;
sigmkp = [0.001,-0.0001];

outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[sigm_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',C,''sigm'',sigmkp);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''sigm'',sigmkp);']);
    outputtable(2*n,1) = {['predict_sigm_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''sigm'',sigmkp);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\results20140212\outputFSVCforAID1608red20140212.xls',outputtable,'sheet4','A1');





%*****************************************************************************************************************%
%*******************************hp1bin6*****20110624**************************************************************%
%*****************************************************************************************************************%
load hp1bin6.mat;

hp1bin61 = hp1bin6(:,[1,2,3,4,5,6,7,9,10,12,14,15]);

[trainset,testset] = partition(hp1bin61,200,200);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);

%********************************Call k-fold cross validation********for hp1bin6*******************************%
%------------------------------linear kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[linear_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',300,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp120110719.xls',outputtable,'sheet1','A1');

%[ks gini] = createksandgini(score_linear_ts1);

%------------------------------polynomial kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[poly_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',10,''poly'',[1,3]);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''poly'',[1,3]);']);
    outputtable(2*n,1) = {['predict_poly_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''poly'',[1,3]);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp120110719.xls',outputtable,'sheet2','A1');

%------------------------------gauss kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[gauss_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',300,''gauss'',0.1);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''gauss'',0.1);']);
    outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''gauss'',0.1);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp120110719.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[sigm_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',10,''sigm'',[0.1,-0.01]);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''sigm'',[0.1,-0.01]);']);
    outputtable(2*n,1) = {['predict_sigm_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''sigm'',[0.1,-0.01]);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp120110719.xls',outputtable,'sheet4','A1');


%*******************************************************************************************************************%
%###################################################################################################################%
%*******************************************************************************************************************%
load hp2bin6.mat;

%*******************************hp2bin6*****20110624**********************************************%
hp2bin61 = hp2bin6(:,[1,2,3,4,5,7,9,10,12,14,15]);

[trainset,testset] = partition(hp2bin61,70,70);
fmst = computeFuzzynumber(trainset,0.001);
[tr1,tr2,tr3,tr4,tr5,fms1,fms2,fms3,fms4,fms5,vl1,vl2,vl3,vl4,vl5] = createcorssvalidationset([trainset fmst],5);


%****************************Call k-fold cross validation********for hp2bin6****************************************%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[linear_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',10,''linear'',[]);']);
    eval(['[predicty_linear_tr' num2str(n) ', predict_linear_tr' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''linear'',[]);']);
    outputtable(2*n,1) = {['predict_linear_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_linear_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_linear_ts' num2str(n) ', predict_linear_ts' num2str(n) '] = testFSVC(linear_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''linear'',[]);']);
    outputtable(2*n+1,1) = {['predict_linear_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_linear_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp220110719.xls',outputtable,'sheet1','A1');

%[ks gini] = createksandgini(score_linear_tr2);

%------------------------------polynomial kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[poly_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',10,''poly'',[1,3]);']);
    eval(['[predicty_poly_tr' num2str(n) ', predict_poly_tr' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''poly'',[1,3]);']);
    outputtable(2*n,1) = {['predict_poly_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_poly_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_poly_ts' num2str(n) ', predict_poly_ts' num2str(n) '] = testFSVC(poly_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''poly'',[1,3]);']);
    outputtable(2*n+1,1) = {['predict_poly_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_poly_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp220110719.xls',outputtable,'sheet2','A1');

%------------------------------guass kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[gauss_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',10,''gauss'',0.1);']);
    eval(['[predicty_gauss_tr' num2str(n) ', predict_gauss_tr' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''gauss'',0.1);']);
    outputtable(2*n,1) = {['predict_gauss_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_gauss_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_gauss_ts' num2str(n) ', predict_gauss_ts' num2str(n) '] = testFSVC(gauss_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''gauss'',0.1);']);
    outputtable(2*n+1,1) = {['predict_gauss_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_gauss_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp220110719.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:5
    eval(['[sigm_tr' num2str(n) ',boundary] = trainFSVC(tr' num2str(n) ',fms' num2str(n) ',10,''sigm'',[0.1,-0.01]);']);
    eval(['[predicty_sigm_tr' num2str(n) ', predict_sigm_tr' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',tr' num2str(n) ',''sigm'',[0.1,-0.01]);']);
    outputtable(2*n,1) = {['predict_sigm_tr' num2str(n)]};
    outputtable(2*n,2:19) = num2cell(eval(['predict_sigm_tr' num2str(n) '(:,1:18)']));
    eval(['[predicty_sigm_ts' num2str(n) ', predict_sigm_ts' num2str(n) '] = testFSVC(sigm_tr' num2str(n) ',boundary,tr' num2str(n) ',testset,''sigm'',[0.1,-0.01]);']);
    outputtable(2*n+1,1) = {['predict_sigm_ts' num2str(n)]};
    outputtable(2*n+1,2:19) = num2cell(eval(['predict_sigm_ts' num2str(n) '(:,1:18)']));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp220110719.xls',outputtable,'sheet4','A1');

%*******************************************************************************************************************%
%###################################################################################################################%
%*******************************************************************************************************************%


%###############################Parameter selection based on Grid search############################################%
load hp1bin6.mat;

%********************************Parameter selection for model*********hp1bin6*********************************************%
%-----------------------hp1bin6----------------------------------------%
hp1bin61 = hp1bin6(:,[1,2,3,4,5,6,7,9,10,12,14,15]);

[trainset,testset] = partition(hp1bin61,200,200);
fms = computeFuzzynumber(trainset,0.001);


%-------------Breastcancer_STD-----------------------%
Breastcancer_STD = minmax(Breastcancer);
[trainset,testset] = partition(Breastcancer_STD,70,70);
fms = computeFuzzynumber(trainset,0.001);


%-------------Diabetes_STD---------------------------%
Diabetes_STD = minmax(Diabetes);
[trainset,testset] = partition(Diabetes_STD,200,200);
fms = computeFuzzynumber(trainset,0.001);


%-------------Heartdisease_STD-----------------------%
Heartdisease_STD = minmax(Heartdisease);
[tr,testset] = partition(Heartdisease_STD,90,90);
fms = computeFuzzynumber(tr,0.001);

%-------------Hepatitis_STD---------------------------%
Hepatitis_STD = minmax(Hepatitis);
[trainset,testset] = partition(Hepatitis_STD,25,25);


%-------------Liverdisorder_STD-----------------------%
Liverdisorder_STD = minmax(Liverdisorder);
[trainset,testset] = partition(Liverdisorder_STD,100,100);
fms = computeFuzzynumber(trainset,0.001);


%-------------Lungcancer_STD-----------------------%
Lungcancer_STD = minmax(Lungcancer);
[trainset,testset] = partition(Lungcancer,7,7);
fms = computeFuzzynumber(trainset,0.001);


%-------------Spectfheartdisease_STD-----------------------%
Spectfheartdisease_STD = minmax(Spectfheartdisease);
[trainset,testset] = partition(Spectfheartdisease_STD,40,40);
fms = computeFuzzynumber(trainset,0.001);


%-------------Spectheartdisease_STD-----------------------%
Spectheartdisease_STD = minmax(Spectheartdisease);
[trainset,testset] = partition(Spectheartdisease_STD,40,40);
fms = computeFuzzynumber(trainset,0.001);


%-------------Wbreastcancer_STD---------------------------%
Wbreastcancer_STD = minmax(Wbreastcancer);
[trainset,testset] = partition(Wbreastcancer_STD,200,200);
fms = computeFuzzynumber(trainset,0.001);


%-------------Aus14std------------------------------------%
[trainset,testset] = partition(aus14std,200,200);
fms = computeFuzzynumber(trainset,0.001);

%-------------Chn20std------------------------------------%
[trainset,testset] = partition(chn20std,200,200);
fms = computeFuzzynumber(trainset,0.001);

%-------------Ger24std------------------------------------%
[trainset,testset] = partition(ger24std,200,200);
fms = computeFuzzynumber(trainset,0.001);

[trainset,testset] = partition(ger30std,200,200);
fms = computeFuzzynumber(trainset,0.001);

%-------------Usa66std------------------------------------%
[trainset,testset] = partition(usa66std,200,200);
fms = computeFuzzynumber(trainset,0.001);

%---------------------------------------------------------%
C1 = [1,8,16,32,64,128,256,512,1024];
C2 = [1,5,10,50,100,500,1000,5000,10000];

%------------------------------linear kernel-------------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [linear_tr,boundary] = trainFSVC(trainset,fms,C2(n),'linear',[]);
    [predicty_linear_ts, predict_linear_ts] = testFSVC(linear_tr,boundary,trainset,testset,'linear',[]);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_linear_ts(:,1:18));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforSPUsa6620110823.xls',outputtable,'sheet1','A1');

%------------------------------polynomial kernel---------------------------%
polykp = [0,2];
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [poly_tr,boundary] = trainFSVC(trainset,fms,C2(n),'poly',polykp);
    [predicty_poly_ts, predict_poly_ts] = testFSVC(poly_tr,boundary,trainset,testset,'poly',polykp);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_poly_ts(:,1:18));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforSPUsa6620110823.xls',outputtable,'sheet2','A25');

%------------------------------gauss kernel--------------------------------%
gausskp = 5;
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [gauss_tr,boundary] = trainFSVC(trainset,fms,C2(n),'gauss',gausskp);
    [predicty_gauss_ts, predict_gauss_ts] = testFSVC(gauss_tr,boundary,trainset,testset,'gauss',gausskp);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_gauss_ts(:,1:18));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforSPUsa6620110823.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel------------------------------%
sigmkp = [0.0001,-0.00001];
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [sigm_tr,boundary] = trainFSVC(trainset,fms,C2(n),'sigm',sigmkp);
    [predicty_sigm_ts, predict_sigm_ts] = testFSVC(sigm_tr,boundary,trainset,testset,'sigm',sigmkp);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_sigm_ts(:,1:18));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforSPUsa6620110823.xls',outputtable,'sheet4','A1');




%********************************Parameter selection for model*********hp2bin6*********************************************%
load hp2bin6.mat;

hp2bin61 = hp2bin6(:,[1,2,3,4,5,7,9,10,12,14,15]);

[trainset,testset] = partition(hp2bin61,70,70);
fms = computeFuzzynumber(tr,0.001);

C1 = [1,8,16,32,64,128,256,512,1024];
C2 = [10,30,50,100,300,500,1000,5000,10000];

%------------------------------linear kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [linear_tr,boundary] = trainFSVC(trainset,fms,C1(n),'linear',[]);
    [predicty_linear_ts, predict_linear_ts] = testFSVC(linear_tr,boundary,trainset,testset,'linear',[]);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_linear_ts(:,1:18));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp22011071901.xls',outputtable,'sheet1','A1');


%------------------------------polynomial kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [poly_tr,boundary] = trainFSVC(trainset,fms,C1(n),'linear',[1,3]);
    [predicty_poly_ts, predict_poly_ts] = testFSVC(poly_tr,boundary,trainset,testset,'linear',[1,3]);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_poly_ts(:,1:18));
 end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp22011071901.xls',outputtable,'sheet2','A1');

%------------------------------guass kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [gauss_tr,boundary] = trainFSVC(trainset,fms,C1(n),'linear',0.1);
    [predicty_gauss_ts, predict_gauss_ts] = testFSVC(gauss_tr,boundary,trainset,testset,'linear',0.1);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_gauss_ts(:,1:18));
end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp22011071901.xls',outputtable,'sheet3','A1');

%------------------------------sigmoid kernel---------------------------%
outputtable(1,:) = {'model','totalg','totalb','gtog','gtob','btob','btog','gerro','berror','totalerror','accuracy','precision','recall','sensitivity','specificity','Fmeasure','correlation','ksscore','giniindex'};

for n=1:1:9
    [sigm_tr,boundary] = trainFSVC(trainset,fms,C1(n),'linear',[0.01,-0.001]);
    [predicty_sigm_ts, predict_sigm_ts] = testFSVC(sigm_tr,boundary,trainset,testset,'linear',[0.01,-0.001]);
    outputtable((n+1),1) = {n};
    outputtable((n+1),2:19) = num2cell(predict_sigm_ts(:,1:18));
 end

xlswrite('D:\FSVCwithMatlab2\outputFSVCforhp22011071901.xls',outputtable,'sheet4','A1');

%*******************************************************************************************************************%
%###################################################################################################################%
%*******************************************************************************************************************%

