image_test_labels = loadLabels('t10k-labels.idx1-ubyte');%添加测试标签
image_test_images = loadImages('t10k-images.idx3-ubyte');%添加测试数据

image_train_labels = loadLabels('train-labels.idx1-ubyte');%添加训练标签
image_train_images = loadImages('train-images.idx3-ubyte');%添加训练数据

%导入训练样本
train_image = zeros(60000, 784);%先为训练样本分配空间
for i = 1 : 60000
  train_A = image_train_images(:, i);%按每一行进行读取
  train_B = reshape(train_A, 28, 28);%转换为28X28矩阵
  train_C = 255 * train_B;%转换为图像
  train_C = reshape(train_C, 1, []);%将矩阵变形
  train_image(i, :) = train_C;%存储图像
end
train_image = double(train_image);

%导入测试样本，步骤同上
test_image = zeros(10000, 784);
for j = 1 : 10000
    test_A = image_test_images(:, j);
    test_B = reshape(test_A, 28, 28);
    test_C = 255 * test_B;
    test_C = reshape(test_C, 1, []);
    test_image(j, :) = test_C;
end
test_image = double(test_image);

%使用TreeBagger来对训练样本进行训练，获得一个model
model = TreeBagger(100,train_image,image_train_labels); 

%之后使用model来对测试样本进行预测，将结果存在result内
result = predict(model,test_image);
result = cell2mat(result); %因为result是cell类的，使用cell2mat转换成字符串

%sc用来保存相减的结果，当其等于0（ASCII里是48）的时候就是识别正确的结果，最终得出识别率
sc = double(result) - image_test_labels; 
count = sum(sc(:) == 48)/100.0;
