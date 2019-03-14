function [I_op] = RGB2opponent(I)
%code based on this thread: https://nl.mathworks.com/matlabcentral/answers/118889-rgb-to-opponent-color-space-code
%color calculator: http://www.easyrgb.com/en/math.php 
%
I = im2douple(I);
I_op = zeros(size(I));

%extract the RGB channels
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

%convert to opponent space
I_op(:, :, 1) = (R - G)./sqrt(2);
I_op(:, :, 2) = (R + G -2*B)./sqrt(6);
I_op(:, : ,3) = (R + G + G)./sqrt(3);

end