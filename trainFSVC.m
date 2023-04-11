%**********************************************************************************************************%
%-------------------------------Create FSVC model for training dataset-------------------------------------%
%------------------------------------Using different kernel function---------------------------------------%
%-------------------------------------------Designed by zhangzw--------------------------------------------%
%------------------------------------------------2011.6.26-------------------------------------------------%
%**********************************************************************************************************%
% /*--------------------------------------------------------------------------------------*/
% /*--------------FSVC model--------------------------------------------------------------*/
% /*--------------Min (1/2)*lamdai*lamdaj*yi*yj*K(xi,xj)-sum(lamdai)----------------------*/
% /*--------------subject to:-------------------------------------------------------------*/
% /*--------------sum(lamdai*yi)=0--------------------------------------------------------*/
% /*--------------0 =< lamdai <= si*C-----------------------------------------------------*/
% /*--------------------------------------------------------------------------------------*/

%---------------------------trainFSVC------------------------------*/
function [lamda,boundary] = trainFSVC(train,fms,C,ker,para)

    %construct objective function
    [nrow,ncol] = size(train);
    
    %find X and Y
    X = train(:,1:ncol-1);
    Y = train(:,ncol);
    
    %objective function with linear part
    f=(-1)*ones(nrow,1);
        
    %define kernel matrix
    Kmatrix = zeros(nrow,nrow);
    Kmatrix = kernel(X,X,ker,para);
    
    a0 = zeros(nrow,1)+0.0001;
    
    %construct Hessian matrix
    H = (Y*Y').*Kmatrix;
   
    %Set LHS of constrains
    Aeq=Y';

    %Set RHS of constrains
    beq=0;
    
    %set boundary of variables
    lb=zeros(nrow,1);
    ub=C*fms;
    
    %Solve the optimal problem
    options = optimset;
    options.LargeScale = 'on';
    options.Display = 'on';
    [x,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,a0,options);
    
    %output the results of classification
    lamda = x;

    %find the boundary of decision hyperplane
    epsilon = 1e-8;                 
    i_sv = find(abs(lamda)>epsilon); 
   
    %[nrow,ncol] = size(train);
   
    %find X and Y
    %X = train(:,1:ncol-1);
    %Y = train(:,ncol);
  
    tmp = kernel(X,X(i_sv,:),ker,para)*(lamda.*Y);         
    b = 1./Y(i_sv)-tmp;
    boundary = mean(b);
    
    

