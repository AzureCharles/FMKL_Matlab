function KMat = kernel(X1,X2,kertype,sigma)
% X1 and X2 are two matrices. Each row of them is a sample.
% type is the type of kernel function.
% K is the kernel matrix.

switch kertype
    case 'linear'
        KMat = X1*X2';
    case 'poly'
        KMat = (X1*X2'+1).^sigma;
    case 'gauss'
        n1 = size(X1,1);
        n2 = size(X2,1);
        KMat = zeros(n1,n2);
        for i = 1:n1
            for j = 1:n2
                KMat(i,j) = exp(-norm(X1(i,:)-X2(j,:))^2/(2*sigma^2));
            end
        end
end