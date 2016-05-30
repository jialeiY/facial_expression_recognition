function [ Isample ] = subsample(I,bboxes,intv1,intv2)
    alpha=0; %control if want to cut more part of the face
    Isample=zeros(intv1,intv2);
    x=bboxes(1,1)+bboxes(1,3)*alpha:bboxes(1,3)*(1-alpha*2)/(intv2-1):bboxes(1,1)+bboxes(1,3)*(1-alpha); %downsample
    y=bboxes(1,2)+bboxes(1,4)*alpha:bboxes(1,4)*(1-alpha*2)/(intv1-1):bboxes(1,2)+bboxes(1,4)*(1-alpha);
    x=round(x);
    y=round(y);
    Isample(:,:)=I(y,x);   
end

