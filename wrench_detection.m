delete(instrfindall);delete(imaqfind);close all;clear all;clc;

path='D:\PhD\Fall 2016\ME 5984 SS Adv Experimental Robotics\Wrench detection\simulationpictures\';
file_name='2.jpg';
d=dir([path file_name]);
c=struct2cell(d);
ii=2;

%% Main Loop

im=imread('2.jpg');
%         figure,imshow(im)
%% Image Cropping
%     img=imcrop(im,[280 577 1280 550]);
[y,x,~]=size(im);
img_hou=imcrop(im,[0 0 x y/2]);
figure(2),subplot(221),imshow(img_hou),title('Region of Interest','FontSize',16)
%% Image Adjust
img2_hou=imadjust(img_hou,stretchlim(img_hou),[]);
img2=imadjust(im,stretchlim(im),[]);

subplot(222),imshow(img2_hou),title('Image Adjust','FontSize',16)
%% BackGround Remove
img_remove_hou=back_ground_remove(img2_hou);
img_remove=back_ground_remove(img2);

subplot(223),imshow(img_remove_hou),title('Background Removal','FontSize',16)
%% Image Segmentation
img_seg_hou=image_segmentation(img_remove_hou);
img_seg=image_segmentation_length(img_remove);

subplot(224),imshow(img_seg_hou),title('Image Segmentation','FontSize',16)
%% Edge Detection
img_edge = edge(img_seg_hou,'canny');
%     subplot(224),imshow(img_edge)
%% Hough transform

color1 =['r','g','b','c','m','y'];
radius =[16 17 18 21 25 28];
size_act=['19';'18';'15';'14';'13';'12'];
%  radius =[13 15 16 19 ];
figure(3),imshow(im);
hold on;

for i=1:length(radius)
  
  [centers,accum_arry] = detectCircles(img_seg_hou, radius(i),1);
  
  %     figure(1);imshow(img_seg)
  % figure(2);imagesc(accum_arry);colormap('jet');
  
  
  plot(centers(:,1),centers(:,2),'+','LineWidth',2,'Color',color1(i));
  for j=1:size(centers,1)
    theta = 0 : 0.01 : 2*pi;
    x = radius(i) * cos(theta) + centers(j,1);
    y = radius(i) * sin(theta) + centers(j,2);
    plot(x, y,color1(i), 'LineWidth', 2);
  end
end

%%  Detect Lenght and Area
color1_l = fliplr(color1);
s=regionprops(img_seg,{'Centroid','BoundingBox','Area'});

numobj=numel(s);
if numobj>0
  
  %         figure(4),imshow(im)
  %         hold on
  Areas=zeros(numobj);
  lenght=zeros(numobj);
  for i=1:numobj
    
    Areas(i)=s(i).Area;
    b_w=s(i).BoundingBox;
    lenght(i)=b_w(4);
  end
  [Areas_ord,ind]=sort(Areas,'descend');
  [lenght_ord,ind_L]=sort(lenght,'descend');
  if length(ind_L)>=6
    mm=6;
  else
    mm=length(ind_L);
  end
  x_length=zeros(1,mm);
  for i=1:mm
    
    cen=s(ind_L(i,1)).Centroid;
    x_length(i)=cen(1);
    
    x=cen(1);
    y=cen(2);
    
    plot(cen(1),cen(2),['h',color1_l(i)],'LineWidth', 3)
    
  end
  
  %% Area
  
  if length(ind)>=6
    mm=6;
  else
    mm=length(ind);
  end
  x_area=zeros(1,mm);
  y_area=zeros(1,mm);
  for i=1:mm
    %     [m,k]=max(Areas);
    
    cen=s(ind(i,1)).Centroid;
    bb=s(ind(i,1)).BoundingBox;
    ar=s(ind(i,1)).Area;
    
    x_area(i)=cen(1);
    y_area(i)=cen(2);
    
    x1=[bb(1) bb(1)+bb(3) bb(1)+bb(3) bb(1) bb(1)];
    y1=[bb(2) bb(2) bb(2)+bb(4) bb(2)+bb(4) bb(2)];
    
    str=num2str(ar);
    
    x=cen(1);
    y=cen(2);
    
    plot(x1,y1, color1_l(i))
    text(((x1(3)+x1(4))/2),y1(2)-10,size_act(i,:),'Color','g','FontSize', 20)
    
  end
end

%% Identify the correct wrench

positions=zeros(4,mm);
positions(1,:)=1:mm;
positions(2,:)=x_area;
positions(3,:)=x_length;

votings=zeros(2,mm);
votings(1,:)=x_area;
votings(2,:)=50;

e=40;  % pixels

for i=1:mm
  v_min=x_length(i)-e;
  v_max=x_length(i)+e;
  if (votings(1,i)>=v_min && votings(1,i)<=v_max)
    votings(2,i)=votings(2,i)+30;
  end
end

voting=votings(2,:);

if (voting(end-2)>=voting(end-1) || voting(end-2)>=voting(end-3))
  
  position(1)=x_area(end-2);
  position(2)=y_area(end-2);
  
elseif (voting(end-1)>=voting(end-2) || voting(end-1)>=voting(end-3))
  
  position(1)=x_area(end-1);
  position(2)=y_area(end-1);
  
elseif (voting(end-3)>=voting(end-2) || voting(end-3)>=voting(end-1))
  
  position(1)=x_area(end-3);
  position(2)=y_area(end-3);
end

text(10,10,['correct wrench x-position is ', num2str(position(1))])








