%**********************************************************************************************************%
%-------------------------Create fuzzy membership table for training instance------------------------------%
%------------------------------------Using MEAN as the representative point--------------------------------%
%**********************************************************************************************************%

%---------------------------computeFuzzynumber------------------------------*/
function fms = computeFuzzyMembership(kernel,label,delta)
    eta = 1e-3; % eta=0.001
    trainset=[kernel label];
    if isempty(trainset) == 1
        disp('The input dataset is null!');
        return;
    else 
        [row1,col1] = size(trainset);
        labels = sort(unique(trainset(:,col1)));
        % label=[l1,l2],size为1*2,输入格式为l1<l2;
        fms = zeros(row1,1); 
        % 初始化模糊系数向量；
        group1 = trainset(trainset(:,col1) == labels(1),:);
        group2 = trainset(trainset(:,col1) ~= labels(1),:);
        % 将训练集分为最小标签类，和非最小标签类；
        % 对于二分类问题，group2等效于第二类别。
        row_g1 = size(group1,1);
        row_g2 = size(group2,1);
        
        mean_g1 = mean(group1(:,1:col1-1));
        mean_g2 = mean(group2(:,1:col1-1));
        
        max_g1 = 0;
        max_g2 = 0;
        % 求范数最大值
        for i=1:1:row_g1
            if sqrt(norm(group1(i,1:col1-1) - mean_g1)) >= max_g1
                max_g1 = sqrt(norm(group1(i,1:col1-1) - mean_g1));
            end
        end
        
        for j=1:1:row_g2
            if sqrt(norm(group2(j,1:col1-1) - mean_g2)) >= max_g2
                max_g2 = sqrt(norm(group2(j,1:col1-1) - mean_g2));
            end
        end
        % 类均值距离改进版
        % 分段函数，delta改为外部定义阈值
        
        % 计算放缩系数
        % exp(-kt)=1-t/(max_g+eta),t=delta
        % k=-log(1-delta/(max_g+eta))/delta
        
        for i=1:row_g1
            if sqrt(norm(group1(i,1:col1-1) - mean_g1)) <= delta
        	    fms(i,1) = 1 - (sqrt(norm(trainset(i,1:col1-1) - mean_g1))/(max_g1 + eta));
   
            else 
                % 范数大于阈值delta时，使用exp(-kt)
                k = log(1-delta/(max_g1+eta))/delta;
    	        fms(i,1) = exp(k*sqrt(norm(trainset(i,1:col1-1) - mean_g1)));
            end
        end
        for j=1:row_g2
            if sqrt(norm(group2(j,1:col1-1) - mean_g2)) <= delta
                fms(j+row_g1,1) = 1 - (sqrt(norm(trainset(j,1:col1-1) - mean_g2))/(max_g2 + eta));
            else
                k = log(1-delta/(max_g2+eta))/delta;
        	    fms(j+row_g1,1) = exp(k*sqrt(norm(trainset(j,1:col1-1) - mean_g2)));
            end
        end
    end