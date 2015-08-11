%%
clear; clc; close all;
cd('C:\Users\vbk4718\Desktop\GC projects\BlackRock');

num_shuffles = 10;
modelAccuracy = zeros(num_shuffles,1);

for i=1:num_shuffles
    fprintf('Shuffle number: %d\n', i);
    modelAccuracy(i) = computeAccuracyBR();
end

fprintf('Mean model accuracy = %f\n', mean(modelAccuracy));
fprintf('Stdev model accuracy = %f\n', std(modelAccuracy));
