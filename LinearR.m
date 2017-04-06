close all;
load('fmri_words.mat');
Data = X_train';
[U, DD, V] = svd(Data, 0);

XX = zeros(1, 276);
YY = zeros(1, 276);
for feat = 25:300
Factor = U(:, 1:feat);
ND = Data'*Factor;
[N, D] = size(ND);

X_new = X_test*Factor;

Y_train_new = zeros(300, 218);
for i = 1:300
Y_train_new(i,:) = word_features_centered(Y_train(i,1), :);
end

K = zeros(N, N);
for i = 1:N
    for j = 1:N
        K(i, j) = (ND(i,:)*ND(j,:)');
    end
end

w = ND'*(inv(K + 10*eye(N)))*Y_train_new;

answer = X_new*w;
count = 0;
for i = 1:60
    true = Y_test(i, 1);
    false = Y_test(i, 2);
    d1 = norm(answer(i, :) - word_features_centered(true, :));
    d2 = norm(answer(i, :) - word_features_centered(false, :));
    if d1 < d2
        count = count + 1;
    end
end
accuracy = count*100/60;
fprintf('features = %d, accuracy = %f\n',feat,accuracy);
XX(feat-24, 1) = feat;
YY(feat-24, 1) = accuracy;
end
plot(XX, YY);