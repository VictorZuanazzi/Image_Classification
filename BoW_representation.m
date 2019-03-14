function [BoW] = BoW_representation(x, sampling_mode, sift_type, vocabulary, display_hist)

[vocab_size, ~] = size(vocabulary);
[~, num_im] = size(x);

BoW = zeros(num_im, vocab_size);

for i = 1:num_im
    d = feature_extraction(x{i}, sampling_mode, sift_type);
    [num, ~] = size(d);
    words = zeros(num, 1);
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
        words(s, 1) = closest;
    end
    
    if display_hist
        figure(i)
    end
    BoW(i,:) = histogram(words, vocab_size, 'Normalization', 'probability');
    
end
end
