function [descriptors] = feature_extraction(I, sampling_strategy, image_type, descriptor_type)
%function to extract the features of an image.
%input:
    %image: an rgb image to have its features extracted.
    %image_type: "gray", "RGB", "opponent".
    %sampling_strategy: "dense" or "key point". 
    %descriptor_type: "sift" or "liop".
%output:
    %descriptors: descriptors of the image.

% Convert image to desired type
grayscale_image = single(rgb2gray(I));
switch image_type
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

switch sampling_strategy
    case "dense"
        f = vl_dsift(grayscale_image, 'Step', 5, 'Size', 21);
    case "key point"
        f = vl_sift(grayscale_image);
    otherwise
        error("Incorrect sampling strategy");
end

% Find descriptors using desired descriptor algorithm
descriptors = [];

for c = 1:channels
    [~, d] = vl_covdet(I(:,:,c), 'Frames', f, 'descriptor', descriptor_type);
    descriptors = [descriptors; d];
end
descriptors = descriptors';
end

