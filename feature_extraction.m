function [descriptors] = feature_extraction(I, mode, sift_type)
%function to extract the features of an image.
%input:
    %image: an rgb image to have its features extracted.
    %mode: "dense" or "key point". 
    %sift_type: "gray", "RGB", "opponent", "None".
%output:
    %descriptors: 128x3 descriptors of the image.

% Convert image to desired type
grayscale_image = single(rgb2gray(I));
switch sift_type
    case "gray"
        I = grayscale_image;
    case "RGB"
        I = single(I);      
    case "opponent"
        I = single(RGB2opponent(I));
    otherwise
        error("Incorrect image type");
end
[~,~,channels] = size(I);

% Perform sift for desired sammpling strategy
descriptors = [];
switch mode
    case "dense"
        for c = 1:channels
            [~, d] = vl_dsift(I(:,:,c));
            descriptors = [descriptors; d];
        end
    case "key point"
        f = vl_sift(grayscale_image);
        for c = 1:channels
            [~, d] = vl_sift(I(:,:,c), 'Frames', f);
            descriptors = [descriptors; d];
        end
    otherwise
        error("Incorrect sampling strategy");
end
descriptors = descriptors';
end

