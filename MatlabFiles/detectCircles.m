function [centers,accum_arry] = detectCircles(im, radius, useGradient)

% I_gray=rgb2gray(im);


[row,col] = size(im);
accum_arry = zeros(row,col);

BW = edge(im,'canny');

if (useGradient==0)
    flag=0;
    [y_ind, x_ind] = find(BW);
else
    flag=1;
   [~,Gdir] = imgradient(BW,'prewitt');
   [y_ind, x_ind] = find(Gdir);
end


for i=1:length(y_ind)
    x_c=x_ind(i);
    y_c=y_ind(i);    
    if (flag==0)
        for ang=1:0.01:2*pi
            x=radius * cos(ang) + x_c;
            y=radius * sin(ang) + y_c;
            
            x=round(x);
            y=round(y);
            if x <= size(accum_arry,2) && x>0 && y <= size(accum_arry,1) && y>0
                accum_arry(y,x)=accum_arry(y,x)+1;
            end
        end        
    else
        x= x_c-radius * cosd(Gdir(y_c,x_c));
        y=y_c+radius * sind(Gdir(y_c,x_c));
        
        x=round(x);
        y=round(y);
        if x <= size(accum_arry,2) && x>0 && y <= size(accum_arry,1) && y>0
            accum_arry(y,x)=accum_arry(y,x)+1;
        end 
    end
end

if radius>10
    num_peaks=1;
else
    num_peaks=5;
end


peaks = houghpeaks(accum_arry,num_peaks);

for i=1:size(peaks,1)
    centers(i,1)=peaks(i,2);
    centers(i,2)=peaks(i,1);
end




