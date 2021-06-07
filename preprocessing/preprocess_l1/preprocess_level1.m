clc
clear all

%addpath('/home/kjakkala/neuralwave/preprocess/preprocess_l1')
%源数据根目录，每一类数据放入一个子目录
srcFolder = '~/data/input';
%输出mat文件目录
destFolder = '~/data/CSI_DATA_INTRUDER_L1';

d = dir(srcFolder);
isub = [d(:).isdir];
classes = {d(isub).name}';
classes(ismember(classes,{'.','..'})) = [];

for i = 1:length(classes)
    d = dir(char(strcat(srcFolder, '/', classes(i))));
    isub = ~([d(:).isdir]);
    files = {d(isub).name};
    mkdir(char(strcat(destFolder, '/', classes(i))));
    
    for j = 1:length(files)
        file = char(strcat(srcFolder, '/', classes(i), '/', files(j)));
        csi_trace = read_csi(file);
        file_name = char(files(j));
        complete_file_name = char(fullfile(destFolder, classes(i), strcat(file_name(1:end-4), '.mat')));
        save (complete_file_name , 'csi_trace')
    end
end
