function fms = computeFuzzyNew(trainset,w,b,delta)

    if isempty(trainset) == 1
        disp('The input dataset is null!');
        return;
    else 
        [row1,col1] = size(trainset);
        %fms = zeros(row1,1);
        group1 = trainset(trainset(:,col1) == 0,:);
        group2 = trainset(trainset(:,col1) == 1,:);
        
        row_g1 = size(group1,1);
        row_g2 = size(group2,1);
        
        mean_g1 = mean(group1(:,1:col1-1));
        mean_g2 = mean(group2(:,1:col1-1));
        
        max_g1 = 0;
        max_g2 = 0; 
        dist=abs(w'*x+b)/sqrt(w'*w);
        distmax=max(dist);
        fms=dist./distmax;

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
        
        
        for i=1:row1
      		if trainset(i,col1) == -1
            	fms(i,1) = 1 - (sqrt(norm(trainset(i,1:col1-1) - mean_g1))/(max_g1 + delta));
            end
            if trainset(i,col1) == 1
            	fms(i,1) = 1 - (sqrt(norm(trainset(j,1:col1-1) - mean_g2))/(max_g2 + delta));
            end
        end
    end