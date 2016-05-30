function [ GaborFeature ] = GaborTrans( G,X,ds)
%in order to accelerate the speed. use fft to transform to frequency
%domain. multiply then ifft back.


[m,n,k]=size(G);
X=fft2(X,m,n); %fft data

GaborFeature=zeros( round(m/ds),round(n/ds),k );

for j=1:k
        temp=abs(ifft2(X.*G(:,:,j))); %multiply the data and filter after fft, then ifft it
        temp=downsample(temp,ds);
        temp=downsample(temp',ds)';
        GaborFeature(:,:,j) = temp;
        %subplot( 5, 8, j ),imshow (  GaborFeature(:,:,j) ,[]); 
end
GaborFeature=GaborFeature(:)'; % convert to one vector
end

