function [ G ] = gaborKernel( x,y,scale_num,orientation_num )
%GABOR Summary of this function goes here
%   Detailed explanation goes here

[X,Y]=meshgrid(-x/2+1:x/2,-y/2+1:y/2); %calculte gabor filter in this area
kmax=pi/2; %parameters
sigma=sqrt(2)*pi;
f=sqrt(2);
G=zeros(x,y,scale_num*orientation_num);%gabor filter
for v=0:scale_num-1 %create filter in 5 scale and 8 orientation
    for u=0:orientation_num-1
        kv=kmax/f^v;
        fi=pi*u/8;
        kuv=[kv*cos(fi);kv*sin(fi)];
        G(:,:,v*8+u+1)=(sum(kuv.^2)/sigma.^2)*exp(-sum(kuv.^2)*(X.^2+Y.^2)/(2*sigma.^2)).*(exp(1i*(kuv(1)*X+kuv(2)*Y))-exp(-sigma^2/2));
        %subplot( 5, 8, v*8+u+1 ),imshow ( real( G(:,:,v*8+u+1) ) ,[]); 
    end
end

end

