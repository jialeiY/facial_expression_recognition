function [ Test_px,Test_class,confusion,Corr ] = KNN( kn,Train,Test,Class,TrainClass )

TestNum=size(Test,1);
ClassNum=6;
n=size(Train,1); %train num
k=ceil(kn(n)); 

Test_px=zeros(ClassNum,TestNum); %compare result for each class
for i=1:TestNum
	dis=sqrt(sum((repmat(Test(i,:),n,1)-Train).^2,2)); %euclidean 
	[~,index]=sort(dis,'ascend'); %distance sort from close to far
	k_small=TrainClass(index(1:k),:);
    for j=1:ClassNum %calculate the number of each class within k
        Test_px(j,i)=sum(k_small==j);
    end      
end

[~,Test_class]=max(Test_px);
confusion=zeros(ClassNum,ClassNum);
for i=1:TestNum
    confusion(Test_class(i),Class(i))=confusion(Test_class(i),Class(i))+1;%confusion matrix
end
Corr=sum(diag(confusion))/TestNum; %accurate rate
end

