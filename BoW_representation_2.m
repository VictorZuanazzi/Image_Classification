function [BoW] = BoW_representation_2(x, sampling_mode, sift_type, vocabulary, display_hist)
%creates the Bag of Word representation of each image.
%input: 
    %x: type cell, containing the data.
    %sampling_mode: "dense" or "key point".
    %sift_type: "gray", "RGB" or "opponent".
    %vocabulary: array with the sift vocabulary 
    %display_hist: bool, if true it displays the hist of all images (use
    %with care)
%output:
    %BoW: array [num_im, vocab_size] with the BoW description for each
    %image.

%get the size of the BoW
[vocab_size, ~] = size(vocabulary);
[~, num_im] = size(x);
BoW = zeros(num_im, vocab_size);

%find the BoW representation for each image
for i = 1:num_im
    
    %extract the relevant features of the image.
    d = feature_extraction(x{i}, sampling_mode, sift_type);
    
    %find the closest vocabulary word for each descriptor of the image.
    [num, ~] = size(d);
    histo = zeros(vocab_size, 1);
    for s = 1:num
        closest = -1;
        closest_distance = Inf;
        for v = 1:vocab_size
            distance = norm(vocabulary(v,:) - double(d(s,:)));
            if distance < closest_distance
                closest = v;
                closest_distance = distance;
            end
        end
        histo(closest, 1) = histo(closest, 1) +1;
    end
    
    %display hist if you feel like it.
    if display_hist
        figure(i)
    end
    
    BoW(i,:) = histo/sum(histo, 'all');
end
end
