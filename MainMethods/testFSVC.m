%**********************************************************************************************************%
%-------------------------------Create kernel matrix for test dataset--------------------------------------%
%------------------------------------Using different kernel function---------------------------------------%
%**********************************************************************************************************%

%---------------------------testFSVC------------------------------*/
function [predictedY,stat] = testFSVC(lamda,boundary,train,test,ker,para)

    [nrow,ncol] = size(train);
    [nrowt,ncolt] = size(test);
    
    X = train(:,1:ncol-1);
    Y = train(:,ncol);
    
    Xt = test(:,1:ncolt-1);
    
    Kmatrix = zeros(nrow,nrowt);
    Kmatrix = kernel(X,Xt,ker,para);
    
    tmp = Kmatrix'*(lamda.*Y);
    py = sign(tmp+boundary);
    
    predictedY = zeros(nrowt,2);
    ActualY = test(:,ncolt);
    predictedY = [ActualY py];
    
%   1totalg 2totalb 3gtog 4gtob 5btob 6btog 7gerro 8berror 9totalerror
%   10accuracy 11precision 12recall 13sensitivity 14specificity	15Fmeasure
%   16correlation 17ksscore 18giniindex
    stat = zeros(1,18);
    for n=1:1:nrowt
        if predictedY(n,1) == -1
           stat(1) = stat(1) + 1;
           if predictedY(n,2) == -1
              stat(3) = stat(3) + 1;
           else
              stat(4) = stat(4) + 1;
           end
        end
        if predictedY(n,1) == 1
           stat(2) = stat(2) + 1;
           if predictedY(n,2) == 1
              stat(5) = stat(5) + 1;
           else
              stat(6) = stat(6) + 1;
           end
        end
    end
    
    stat(7) = stat(4) / (stat(3) + stat(4));
    stat(8) = stat(6) / (stat(5) + stat(6));
    stat(9) = (stat(4) + stat(6)) / (stat(3) + stat(4) + stat(5) + stat(6));
    stat(10) = (stat(3) + stat(5)) / (stat(3) + stat(4) + stat(5) + stat(6));
    stat(11) = stat(5) / (stat(4) + stat(5));
    stat(12) = stat(5) / (stat(5) + stat(6));
    stat(13) = stat(5) / (stat(5) + stat(6));
    stat(14) = stat(3) / (stat(3) + stat(4));
    stat(15) = 2*stat(5) / (2*stat(5) + stat(4) + stat(6));
    stat(16) = (stat(3)*stat(5) - stat(4)*stat(6)) / sqrt((stat(5) + stat(6))*(stat(4) + stat(5))*(stat(3) + stat(4))*(stat(3) + stat(6)));
    
    %-------------commented these codes on Jan. 25, 2012-------------------%
%    %ks and gini
%   target = [];
%
%   temp = [tmp+boundary ActualY];
%   temp(temp(:,2) == -1,2) = 0;
%
%   [trow tcol] = size(temp);
%   for n=1:trow
%       temp(n,1) = 1/(1+exp(-temp(n,1)));
%   end
%
%   temp = sortrows(temp,1);
%
%   %g1no at col1, g2no at col2, g1per at col3, g2per at col4, bothper at col5, ks at col6
%   target = zeros(trow,7);
%
%   if temp(1,2) == 1
%       target(1,1) = 1;
%       target(1,2) = 0;
%   else
%       target(1,2) = 1;
%       target(1,1) = 0;
%   end
%
%   for n=2:trow
%       if temp(n,2) == 1
%           target(n,1) = target(n-1,1) + 1;
%           target(n,2) = target(n-1,2);
%       else
%           target(n,2) = target(n-1,2) + 1;
%           target(n,1) = target(n-1,1);
%       end
%   end
%
%   target(:,3) = target(:,1)./max(target(:,1));
%   target(:,4) = target(:,2)./max(target(:,2));
%   target(:,5) = (target(:,1) + target(:,2))./(max(target(:,1)) + max(target(:,2)));
%   target(:,6) = abs(target(:,3) - target(:,4));
%
%   %add to origin of coordinates
%   target = [0,0,0,0,0,0,0; target];
%
%   target(1,7) = (target(1,3) * target(1,4))/2;
%   for n=2:trow
%       target(n,7) = target(n-1,7) + ((target(n,3)-target(n-1,3)) * (target(n,4)+target(n-1,4)))/2;
%   end
%
%   %ks score
%   stat(17) = max(target(:,6));
%   %gini index
%   %stat(18) = 2*max(target(:,7)) - 1;
%   %AUC
%   stat(18) = max(target(:,7));
    %-------------------------------------------------------------%   
    patterns = [];                       
    patterns = predictedY;           
    patterns = sortrows(patterns,-1);
    yy = patterns(:,2);              
    p = cumsum(yy==1);               
    tp = p/sum(yy==1);               
    m = cumsum(yy==-1);              
    fp = m/sum(yy==-1);              
                                     
    nn = length(tp);                 
    YY = (tp(2:nn) + tp(1:nn-1))/2;  
    XX = fp(2:nn) - fp(1:nn-1);      
    ZZ = abs(tp(2:nn) - fp(1:nn-1)); 
    % ks score                       
    stat(17) = max(ZZ);              
    % auc=sum(YY.*XX)-0.5;           
    % auc=sum(YY.*XX);               
    stat(18) = sum(YY.*XX);        
      