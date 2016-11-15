function img2=image_segmentation_length(img1)

  % r=img1(:,:,1);
  % g=img1(:,:,2);
  % b=img1(:,:,3);
  %
  % r=r-mean(mean(r));
  % g=g-mean(mean(g));
  % b=b-mean(mean(b));
  %
  % img2(:,:,1)=r;
  % img2(:,:,2)=g;
  % img2(:,:,3)=b;
  img2=rgb2gray(img1);
  img2=medfilt2(img2);

  level = graythresh(img2); %0.5
  final = im2bw(img2,level);


  remove=bwareaopen(final,3);
  remove=imcomplement(remove);
  se=strel('disk',10);
  disk=imclose(remove,se);
  [~,img2]=bwboundaries(disk,'noholes');
  %     img2=imcomplement(img2);

end

   