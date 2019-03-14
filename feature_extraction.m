function [descriptors] = feature_extraction(I, mode, sift_type)
%function to extract the features of an image.
%input:
    %image: an rgb image to have its features extracted.
    %mode: "dense" or "key point". 
    %sift_type: "gray", "RGB", "opponent", "None".
%output:
    %descriptors: 128x3 descriptors of the image.
    
    

    
switch sift_type
    case "gray"
        I = single(rgb2gray(I));
        descriptors = feature_extraction(I, mode, "None");
        %descriptors = [descriptors; descriptors; descriptors];
        
    case "RGB"
        I = single(I);
        descriptors = [feature_extraction(I(:,:,1), mode, "None");
            feature_extraction(I(:,:,2), mode, "None");
            feature_extraction(I(:,:,3), mode, "None")];
        
    case "opponent"
        I = single(RGB2opponent(I));
        descriptors = [feature_extraction(I(:,:,1), mode, "None");
            feature_extraction(I(:,:,2), mode, "None");
            feature_extraction(I(:,:,3), mode, "None")];
        
end

    
    
switch mode
    case "dense"
        %vl_dsift only does gray scale, single precision
        %mathworks starting point: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/29800/versions/3/previews/reco_toolbox/html/demo_denseSIFT.html
    case "key point"
        %vl_sift only does gray scale, single precision
        
end

    
end

