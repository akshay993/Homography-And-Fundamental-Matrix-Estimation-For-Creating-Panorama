%Stiching of Images
%Created By Akshay Chopra
%Person Number: 50248989
%Email Id: achopra6@buffalo.edu


image1=imread('../data/part1/ledge/1.jpg');
image2=imread('../data/part1/ledge/2.jpg');


image1rgb=image1; %Storing the rgb image1
image2rgb=image2; %Storing the rgb image2
 
%image1=histeq(image1);  %Used only for UTTOWER Image
%image2=histeq(image2);  %Used only for UTTOWER Image

inliers_list=zeros(1000,1);


%Converting images to GrayScale
image1=rgb2gray(image1);
image2=rgb2gray(image2);

%If Images are not floating point type, we convert them
if(~isfloat(image1))
  image1=im2double(image1);
end


%If Images are not floating point type, we convert them
if(~isfloat(image2))
  image2=im2double(image2);
end


%Feature point detection using Harris Detector
[cim1, r1, c1] = harris(image1, 3, 0.05,2 ,0); 

[cim2, r2, c2] = harris(image2, 3, 0.05,2 ,0 );


%Calculating Neighbours of the feauture points of image 1 and image 2
temp = [];
temp2 = [];

size_neighbourhood =9; %Setting the neighbourhood size
pad_size = floor(size_neighbourhood/2); %Calculating padding based on the neighbourhood size
image1 = padarray(image1,[pad_size pad_size],'replicate');
image2 = padarray(image2,[pad_size pad_size],'replicate');

for i= 1:size(r1,1)    
   x = r1(i) + pad_size;
   y = c1(i) + pad_size;
   neighbourhood_1 = image1((x- pad_size):(x + pad_size), (y- pad_size):(y + pad_size));
   descriptor_1 = reshape(neighbourhood_1, [1,size_neighbourhood.^2]);
   descriptor_1= cat(1, temp, descriptor_1);
   temp = descriptor_1;  
end


for i= 1:size(r2,1)
   x = r2(i)+pad_size;
   y = c2(i)+pad_size;
   neighbourhood_2 = image2((x- pad_size):(x + pad_size), (y - pad_size):(y + pad_size));
   descriptor_2 = reshape(neighbourhood_2, [1,size_neighbourhood.^2]);
   descriptor_2= cat(1, temp2, descriptor_2);
   temp2 = descriptor_2;  
end


%Calculating Distance between descriptors 1 and 2
n2=dist2(descriptor_1,descriptor_2);

%Selecting matches with distance lower than threshold
[matches_img1,matches_img2]=find(n2()<0.5);

%Creating matches matrix which contain coordinates of matches from image 1 and image 2
matches=[ c1(matches_img1) r1(matches_img1) c2(matches_img2) r2(matches_img2)];



%Next 4 lines (which are commented) display matches in image1 and image2
% imshow([image1rgb image2rgb]); hold on;
% plot(matches(:,1), matches(:,2), '+r');
% plot(matches(:,3)+size(image1,2), matches(:,4), '+r');
% line([matches(:,1) matches(:,3) + size(image1,2)]', matches(:,[2 4])', 'Color', 'r');



%Running RANSAC and finding homography matrix
%Running for 1000 iterations
for j=1 : 1000
        %Selecting 4 random points
        randpts=randperm(size(matches,1),4);
        randpts=randpts';
        
        %coordinates of matches from Image1
        xx = matches(randpts,1)
        yy = matches(randpts,2)

        %Corresponding coordinates of matches from Image 2
        XX =matches(randpts,3);
        YY = matches(randpts,4);

        A=zeros(8,9);

        xx=xx';
        yy=yy';
        XX=XX';
        YY=YY';
        
        %Calculating the A matrix 
        for i=1:4
                    
            %For calculating A Matrix, taken Reference from youtube video
            %Link: https://www.youtube.com/watch?v=hEz_rYN57Co
            A(2*i-1,:)=[xx(i),yy(i),1,0,0,0,-xx(i)*XX(i),-XX(i)*yy(i),-XX(i)];
            A(2*i,:)=[0,0,0,xx(i),yy(i),1,-xx(i)*YY(i),-YY(i)*yy(i),-YY(i)];
        
        end        
        
        %Now using svd method for calculating homography matrix
        [U,S,V]=svd(A);
        h=V(:,end);
        
        H=reshape(h,3,3)';
        H = H/H(3,3);

        im1=ones(3,1);
        im2=ones(3,1);
        imgnew=ones(3,1);

        inliers=0;
        
        %Running homography matrix on all coordinates in 'matches' matrix
        for icounter=1: size(matches,1)
            im1=[matches(icounter,1);matches(icounter,2);1];
            im2=[matches(icounter,3);matches(icounter,4);1];                       
            imgnew=H*im1;
            imgnew = imgnew/imgnew(3);
            
            %Calculating ssd between coordinates found using homography
            %matrix and original coordinates in 2nd image
            diffrnce = imgnew - im2;
            ssd = sum(diffrnce(:).^2);
            %ssd=sqrt(ssd);
            
            if ssd < 10 %Using threshold as 10 to find whether it is an inlier or not
                inliers = inliers +1;
                distance_resi=pdist([imgnew(1),imgnew(2);im2(1),im2(2)],'euclidean');
                
            end
            
        end
        
        inliers_list(j,1)=inliers; %Creating inliers list for each iteration
        
        if j==1
            homography_list=H;  %Creating homography list for each iteration   
            
        else
            homography_list=cat(3,homography_list,H); %Creating homography list for each iteration
            
        end
        
        
end

[max_inlier,index_max]=max(inliers_list);   %Finding the maximun inliers

best_homography=homography_list(:,:,index_max); %Selecting the homography matrix corresponding to the maximum inliers
A= best_homography';


%Stitching the images to form panorama

T = maketform('projective',A);

[left,x_range,y_range]=imtransform(image1rgb,T, 'nearest');

xdataout=[min(1,x_range(1)) max(size(image2rgb,2),x_range(2))];

ydataout=[min(1,y_range(1)) max(size(image2rgb,1),y_range(2))];

left= imtransform(image1rgb,T,'nearest','XData',xdataout,'YData',ydataout);
final_image = left;

[height, width, three_dim] = size(left);

total_size=height*width*three_dim;

right= imtransform(image2rgb,maketform('affine',eye(3)),'nearest','XData',xdataout,'YData',ydataout);

for i = 1:total_size
    
    if(final_image(i) == 0)
        final_image(i) = right(i);
    elseif(final_image(i) ~= 0 && right(i) ~= 0)
        final_image(i) = left(i)/2 + right(i)/2;
    end
    
end

%Final stiching image
imshow(final_image);
