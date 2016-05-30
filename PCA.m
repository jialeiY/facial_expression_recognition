function [ pca_face,W,m ] = PCA( Data )
%PCA Summary of this function goes here
%   Detailed explanation goes here

m=mean(Data);
t=bsxfun(@minus,Data,m); %data minus mean
sigma=t*t'/(size(Data,1)-1); %covariance
[V, D]=eig(sigma); %eigen value
D=diag(D);
[D,index]=sort(D,1,'descend'); %sort the eigen value from large to small
V=V(:,index);
for i=1:length(D)
    if sum(D(1:i))/sum(D)>=0.9 %choose 90% eigenvalue represent the original data
        break;
    end
end
V=V(:,1:i); %corresponding eigenvector
D=D(1:i)';
W=t'*V./repmat(sqrt(D),size(Data,2),1); %create transform matrix
pca_face=W'*t'; %apply to face

end

