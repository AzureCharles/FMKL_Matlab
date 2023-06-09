% 基于支持超平面距离的模糊隶属度计算
% dist = abs(w'*x+b)/sqrt(w'*w);
% distmax = max(dist);
% fms = dist./distmax;

function fms = computeFuzzyNew(trainSetKernel,w,b)
    eta = 1e-3; % eta=0.001
    if isempty(trainSetKernel) == 1
        disp('The input dataset is null!');
        return;
    else 
        [row1,col1] = size(trainSetKernel);
        labels = sort(unique(trainSetKernel(:,col1)));

        fms = zeros(row1,1);
        group1 = trainSetKernel(trainSetKernel(:,col1) == labels(1),:);
        group1Ex = trainSetKernel(trainSetKernel(:,col1) ~= labels(1),:);
        
        class1Num = size(group1,1);
        class1NumEx = size(group1Ex,1);
        % 将训练集分为最小标签类，和非最小标签类；
        % 对于二分类问题，group1Ex等效于第二类别。
        dist1 = abs(w'*group1+b.*ones(class1Num,1))/sqrt(norm(w));
        dist2 = abs(w'*group1Ex+b.*ones(class1NumEx,1))/sqrt(norm(w));
        
        maxDist1 = max(dist1);
        maxDist2 = max(dist2); 
        
        for i=1:row1
            if trainSetKernel(i,col1) == labels(1)
            	fms(i,1) = dist1(i)/(maxDist1 + eta);
            else
            	fms(i,1) = dist2(i)/(maxDist2 + eta);
            end
        end
    end