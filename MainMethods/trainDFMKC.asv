%**********************************************************************************************************%
%-------------------------------Create FSVC model for training dataset-------------------------------------%
%------------------------------------Using different kernel function---------------------------------------%
%**********************************************************************************************************%
% /*--------------------------------------------------------------------------------------*/
% /*--------------FSVC model--------------------------------------------------------------*/
% /*--------------Min (1/2)*lamdai*lamdaj*yi*yj*K(xi,xj)-sum(lamdai)----------------------*/
% /*--------------subject to:-------------------------------------------------------------*/
% /*--------------sum(lamdai*yi)=0--------------------------------------------------------*/
% /*--------------0 =< lamdai <= si*C-----------------------------------------------------*/
% /*--------------------------------------------------------------------------------------*/
function [model,net] = trainDFMKC(x,y,nLayers,LR,maxI,C)
% Deep Fuzzy Multiple Kernel Learning by Span Bound
% 
% Inputs:
% (1) x = trainng data matrix, where rows are instances and columns are features
% (2) y = training target vector, where rows are instances
% (3) nLayers = number of layers, 1, 2 or 3
% (4) LR = learning rate (default=1E-4)
% (5) maxI = maximum number of iterations (default=100)
% (6) C = SVM penalty constant (default=10)
% fms:calculated during algorithm
%
% Outputs:
% (1) model = LIBSVM model
% (2) net = net parameters

%default values
SetDefaultValue(4,'LR',1E-4);
SetDefaultValue(5,'maxI',100);
SetDefaultValue(6,'C',10);

%initialize weights
betas = ones(nLayers,4)./4;

%initialize kernels
dotx = x*x';
sig = DetermineSig(dotx);
[~,Kf] = computeKernels(dotx,sig,betas,nLayers);

% calulate
%alternating opt
[r,~] = size(x);
span = 0;
for t=1:maxI
    
    %kernels
    [K,Kf] = computeKernels(dotx,sig,betas,nLayers);
    
    w=betas;
    
    % 计算距离
    dist=abs(w'*x+boundary)/sqrt(w'*w);
    distmax=max(dist);
    fmst=dist./distmax;

    %train SVM
    Ks = reshape(Kf(:,nLayers),r,r);
    model = svmtrain(y, [(1:r)',Ks], ['-c C*fmst -t 4 -q 1']);

    %span gradient
    if nLayers==1,
        [betas,spanT] = grad1Layer(model,betas,LR,Kf,K,y);
    elseif nLayers==2,
        [betas,spanT] = grad2Layer(model,betas,LR,Kf,K,sig,y);
    elseif nLayers==3,
        [betas,spanT] = grad3Layer(model,betas,LR,Kf,K,sig,y);
    end

    %feasible region projection
    betas(find(betas<0))=0; %non-negative
    if sum(betas(end,:))>1,
        betas(end,:) = betas(end,:)./sum(betas(end,:)); %trace final layer upper bound
    end
    
    %stopping conditions
    if isnan(sum(betas)),
        error('myApp:argChk', 'Learning rate is too high');
    elseif abs(span-spanT)<1E-4 && t>5,
        break;
    end
    span=spanT;
    
end

%final model
net.w = betas;
net.sig = sig;
net.nLayers = nLayers;
net.n = r;



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
    % Kmatrix = kernel(X,X,ker,para);
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
  
    tmp = kernel(X,X(i_sv,:),ker,para)'*(lamda.*Y);         
    b = 1./Y(i_sv)-tmp;
    boundary = mean(b);



    
    

