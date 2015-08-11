function[Accuracy] = computeAccuracyBR()

%%
% fprintf('Reading data...; ');
data = readtable('datasetBR.csv');
m = round(0.9*size(data,1));
idx = randperm(size(data,1))';
xT1 = table2array(data(idx(1:m),1:end-2));
temp = char(table2array(data(idx(1:m),end-1)));
xt1 = str2num(temp(:,1:4));
xTrain = [xT1, xt1];
yTrain = table2cell(data(idx(1:m),end));
xT2 = table2array(data(idx(m+1:end),1:end-2));
temp = char(table2array(data(idx(m+1:end),end-1)));
xt2 = str2num(temp(:,1:4));
xTest = [xT2, xt2];
yTest = table2cell(data(idx(m+1:end),end));

xTrain(:,1) = nominal(xTrain(:,1));
xTrain(:,2) = nominal(xTrain(:,2));
xTest(:,1) = nominal(xTest(:,1));
xTest(:,2) = nominal(xTest(:,2));

%% Model construction
% fprintf('Model construction...; ');

cTree = ClassificationTree.template('CategoricalPredictors',[1 2]);
cFit = fitensemble(xTrain,yTrain,'Bag',100,cTree,'type','classification');

[yLabels,~] = predict(cFit,xTest);

%% Performance
% fprintf('Post processing...\n');
count1 = 0;
count2 = 0;
for i=m+1:size(data,1)
    count1 = count1 + 1;
    if(strcmp(yLabels(count1),yTest(count1))==1)
        count2 = count2 + 1;
    end
end
Accuracy = count2 * 100 / count1;
