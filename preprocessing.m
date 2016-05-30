function [Face,Mouth,Eyes ] = preprocessing( flist,data_path,emo,intv_face,intv_eyes,intv_mouth )
%REPROCESSING Summary of this function goes here
%   Detailed explanation goes here

n=length(flist); %number of input image
intv2_face=prod(intv_face); 
intv2_eyes=prod(intv_eyes);
intv2_mouth=prod(intv_mouth);

Face=zeros(n,intv2_face); %face
Mouth=zeros(n,intv2_mouth);%32*64
Eyes=zeros(n,intv2_eyes);%128*32

%%%%%%%%%%build object detector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DetectorName={'FrontalFaceCART';'LeftEye';'RightEye';'Mouth'};
mins=[120 120;round(max(intv_face)/5)*ones(3,2)];
mins_req=[20 20;12 18;12 18;15 25];
for i=1:4
    FacialPartsDetect.detector{i}=vision.CascadeObjectDetector(DetectorName{i},'MergeThreshold',4,'MinSize',max(mins_req(i,:),mins(i,:)));
end
bboxes=zeros(size(Face,1),4,3);

for fr=1:n
    I=imread(fullfile(data_path,num2str(emo),flist(fr).name)); %read in image
    if size(I,3)==3 % if rgb convert to graylevel
        I=rgb2gray(I);
    end
    %%%%%%%%%%%  face detect  %%%%%%%%%%%%%
    bbf=step(FacialPartsDetect.detector{1}, I);
    numf=size(bbf,1); %number of face detected
    num_part=zeros(numf,1); %how many facial parts are detected on the possible face 
    Isample=zeros(numf,intv2_face); %face vector
    bt=zeros(numf,12); %every row presents the bbox for face, left, right eyes, mouths
    for num=1:numf %every possible face
        im=subsample(I,bbf(num,:),intv_face(1),intv_face(2)); %subsample resize the face size
        im=histeq(uint8(im));
        Isample(num,:)=reshape(im,1,intv2_face);%convert matrix to vector
        %%%%%%%%%%%%% facial parts detect : mouth and eyes%%%%%%%%%%%%%%%%
            for i=2:4
                switch i
                    case 2
                        region = [1 round(intv_face(1)/1.5); 1 round(intv_face(2)/1.5)]; %left up part of face
                    case 3
                        region = [1 round(intv_face(1)/1.5); round(intv_face(2)/3) intv_face(2)]; %right up part
                    case 4
                        region = [round(intv_face(1)/3) intv_face(1);1 intv_face(2) ];% bottom part
                end             
                im_region= uint8(im(region(1,1):region(1,2),region(2,1):region(2,2))); %regional face
                bp=step(FacialPartsDetect.detector{i}, im_region);% detect facial parts
                if size(bp>0,1) %facial parts exist
                    num_part(num)=num_part(num)+1;%increase num
                    switch i %keep the most possible result
                        case 2 %left eye
                            [~,index]=min(bp(:,1));%left one
                            bp=bp(index,:);                           
                        case 3 %right eye
                            [~,index]=max(bp(:,1)); %right one
                            bp=bp(index,:);                           
                        case 4 %mouth
                            [~,index]=max(bp(:,2)); %bottom one
                            bp=bp(index,:);    
                    end
                   bp(1) = bp(1)+region(2,1)-1; %convert to whole face region
                   bp(2) =  bp(2)+ region(1,1)-1 ;
                   bp(3) = bp(3);
                   bp(4) =  bp(4) ;

                   bt(num,1+(i-2)*4:4+(i-2)*4) = bp;
                end
            end            
    end
    
    if sum(num_part==3)~=0 
        bt=bt(num_part==3,:);%keep the one find all three facial parts
        Isample=Isample(num_part==3,:);            
        bboxes(fr,:,1)=bbf(num_part==3,:); %face
        bboxes(fr,:,2)=[bt(1:2),bt(5:6)-bt(1:2)+bt(7:8)]; %combine two eyes part to one   
        bboxes(fr,:,3)=bt(9:12); %mouth
        Face(fr,:)=Isample;
        Mouth(fr,:)=reshape(subsample(im,bboxes(fr,:,3),intv_mouth(1),intv_mouth(2)),1,intv2_mouth);
        Eyes(fr,:)=reshape(subsample(im,bboxes(fr,:,2),intv_eyes(1),intv_eyes(2)),1,intv2_eyes);
    end
end



end

