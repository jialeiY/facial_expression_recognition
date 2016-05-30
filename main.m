clear;
clc;

intv_face=[128,128]; % face 128*128
intv_eyes=[32,128]; %eyes 32*128
intv_mouth=[32,64]; %mouth 32*64
intv2=prod(intv_face);
intv2_eyes=prod(intv_eyes);
intv2_mouth=prod(intv_mouth);
data_path='.\dataset2\';
a=0.7; %training data percentage

Train=cell(1,3);%face mouth eyes
Test=cell(1,3);
Face=cell(1,6);
Mouth=cell(1,6);
Eyes=cell(1,6);
data_num=zeros(1,6); %overall data inclue training and testing. Every data can detect face, eyes and mouth
train_num=zeros(1,6);
test_num=zeros(1,6);
Train_Class=[];
Test_Class=[];

for emo=1:6
flist=dir(fullfile(data_path,num2str(emo),'*.png'));

[Face{emo},Mouth{emo},Eyes{emo} ] = preprocessing( flist,data_path,emo,intv_face,intv_eyes,intv_mouth );%detect face, facial parts
p =(sum(Face{emo},2) == 0); %delete no detected 
Face{emo}(p,:) = [];
Mouth{emo}(p,:) = [];
Eyes{emo}(p,:) = [];

data_num(emo)=size(Face{emo},1);%create training set
train_num(emo)=round(a*data_num(emo)); 
Train_Class=cat(1,Train_Class,emo*ones(train_num(emo),1));
Train{1}=[Train{1};Face{emo}(1:train_num(emo),:)];
Train{2}=[Train{2};Mouth{emo}(1:train_num(emo),:)];
Train{3}=[Train{3};Eyes{emo}(1:train_num(emo),:)];

test_num(emo)=data_num(emo)-train_num(emo); %create testing set
Test_Class=cat(1,Test_Class,emo*ones(test_num(emo),1));
Test{1}=[Test{1};Face{emo}(train_num(emo)+1:end,:)];
Test{2}=[Test{2};Mouth{emo}(train_num(emo)+1:end,:)];
Test{3}=[Test{3};Eyes{emo}(train_num(emo)+1:end,:)];
end

save('Test.mat','Test');
clear Mouth Eyes Test;

%%

%%%%%%% PCA %%%%%%%%%%%%%%%%%%
[pca_face,w_face,pca_m_face]=PCA(Train{1});

%%%%%%%%%%%Gabor%%%%%%%%%%%%%%
x=64; %Gabor kernel size
y=64;
scale_num=5; %5 scale
orientation_num=8; %8 orientation
ds=2;%down sample rate
G= gaborKernel( x,y,scale_num,orientation_num ); %kernel function
g_mouth=fft2(G,x+intv_mouth(1)-1,y+intv_mouth(2)-1);%fft gabor kernel
g_eyes=fft2(G,x+intv_eyes(1)-1,y+intv_eyes(2)-1);
G_mouth=zeros(sum(train_num),round((x+intv_mouth(1)-1)/ds)*round((y+intv_mouth(2)-1)/ds)*scale_num*orientation_num); %allocate memory: store the gabor feature of
G_eyes=zeros(sum(train_num),round((x+intv_eyes(1)-1)/ds)*round((y+intv_eyes(2)-1)/ds)*scale_num*orientation_num);    %mouth and eyes

for i=1:sum(train_num)  %apply  gabor transoform
    X=reshape(Train{2}(i,:),intv_mouth);
    G_mouth(i,:) = GaborTrans( g_mouth,X,ds );
     X=reshape(Train{3}(i,:),intv_eyes);
    G_eyes(i,:) = GaborTrans( g_eyes,X,ds );
end

%%%%pca reduce dimension for gabor features %%%%%%%%%%%%%%%%
[pca_mouth,w_mouth,pca_m_mouth]=PCA(G_mouth);
[pca_eyes,w_eyes,pca_m_eyes]=PCA(G_eyes);

clear Train G_mouth G_eyes;

Train= [pca_face',5*pca_mouth',5*pca_eyes']; %build the feature vector, mouth and eyes have more weight
%%%%%%%%%% apply to test data %%%%%%%%%%%%%%%%%%%%%%%
load Test;
%pca for face
t=bsxfun(@minus,Test{1},pca_m_face);
Test_face=w_face'*t';

%gabor
G_mouth=zeros(sum(test_num),round((x+intv_mouth(1)-1)/ds)*round((y+intv_mouth(2)-1)/ds)*scale_num*orientation_num);
G_eyes=zeros(sum(test_num),round((x+intv_eyes(1)-1)/ds)*round((y+intv_eyes(2)-1)/ds)*scale_num*orientation_num);
for i=1:sum(test_num)
    X=reshape(Test{2}(i,:),intv_mouth);
    G_mouth(i,:) = GaborTrans( g_mouth,X,ds );
    X=reshape(Test{3}(i,:),intv_eyes);
    G_eyes(i,:) = GaborTrans( g_eyes,X,ds );
end
%pca for gabor
t=bsxfun(@minus,G_mouth,pca_m_mouth);
Test_mouth=w_mouth'*t';

t=bsxfun(@minus,G_eyes,pca_m_eyes);
Test_eyes=w_eyes'*t';

clear Test G_mouth G_eyes;
Test= [Test_face',5*Test_mouth',5*Test_eyes'];
%%%%%%%%%%%%%classfication KNN %%%%%%%%%%%%%%%%%%

kn=@(n) sqrt(n);
[px,test_results,confusion,Corr ] = KNN( kn,Train,Test,Test_Class,Train_Class );










