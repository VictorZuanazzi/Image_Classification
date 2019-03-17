function [descriptors] = feature_extraction(I, sampling_strategy, image_type, descriptor_type)
%function to extract the features of an image.
%input:
    %image: an rgb image to have its features extracted.
    %image_type: "gray", "RGB", "opponent".
    %sampling_strategy: "dense" or "key point". 
    %descriptor_type: 'sift' or 'liop'.
%output:
    %descriptors: descriptors of the image.

% Dense is only available for sift.
if sampling_strategy == "dense" && ~strcmp(descriptor_type, 'sift')   
    error("dense only available for sift");
end
    
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

% Handle desired sampling strategy
descriptors = [];
switch sampling_strategy
    case "dense"
        for c = 1:channels
            [~, d] = vl_dsift(grayscale_image, 'Step', 5, 'Size', 21);
        descriptors = [descriptors; d];
        end
    case "key point"
        f = vl_sift(grayscale_image);
        % Find descriptors using desired descriptor algorithm
        for c = 1:channels
            [~, d] = vl_covdet(I(:,:,c), 'Frames', f, 'descriptor', descriptor_type);
            descriptors = [descriptors; d];
        end
    otherwise
        error("Incorrect sampling strategy");
end

descriptors = descriptors';
end

