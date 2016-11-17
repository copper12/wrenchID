function I2=back_ground_remove(I)

  [counts,x]=imhist(I(:,:,1));
  [m1,i1]=max(counts);

  [counts,x]=imhist(I(:,:,2));
  [m2,i2]=max(counts);

  [counts,x]=imhist(I(:,:,3));
  [m3,i3]=max(counts);
  % i1=200;
  % i2=250;
  % i3=185;

  r2=I(:,:,1)+0.5*i1;
  g2=I(:,:,2)+0.5*i2;
  b2=I(:,:,3)+0.5*i3;

  I2=I;
  I2(:,:,1)=r2;
  I2(:,:,2)=g2;
  I2(:,:,3)=b2;
  i=[i1 i2 i3];

end