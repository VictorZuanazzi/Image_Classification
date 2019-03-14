function[x, y] = load_n_reshape(file_path, keep, im_dim)
%loads the dataset in the standford format
%input:
    %file_path: string with the path to the file.
    %keep: array of size 5 containing the classes we want to keep.
    %im_dim: array [width, height, channels]
%output:
    %x: cell of images in the specified dimensions in im_dim
    %y: vector containing the labels. The ith index of y corresponds to the
    %ith image in x.
    
%load the training data.
D = load(file_path);

%only keep the classes of interest.
keep_idx = find(D.y == keep(1) | D.y == keep(2) | D.y == keep(3) | D.y == keep(4) | D.y == keep(5));
x_t = D.X(keep_idx, :);
y = D.y(keep_idx);

%reshapes all images.
[num_im, ~] = size(x_t);
for i = 1:num_im
    x{i} = reshape(x_t(i,:), im_dim);
end

end