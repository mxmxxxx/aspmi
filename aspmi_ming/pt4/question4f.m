%% Q4.6 - Deep network vs simple dynamical perceptron
% MATLAB implementation for nonlinear prediction comparison
% Compares:
%   1) Simple dynamical perceptron
%   2) Deep neural network with 4 hidden layers
%
% Default parameters:
%   - 4 hidden layers
%   - 20,000 epochs
%   - learning rate = 0.01
%   - noise power = 0.05

clear; close all; clc;
rng(1);

%% ------------------------------------------------------------
% 1. Generate a highly nonlinear signal
%% ------------------------------------------------------------
N = 2000;
n = (1:N)';

% Input signal x(n): sum of nonlinear sinusoidal components
x = 1.2*sin(2*pi*0.003*n) + ...
    0.9*cos(2*pi*0.011*n + 0.6) + ...
    0.6*sin(2*pi*0.021*n.^1.02) + ...
    0.5*cos(2*pi*0.035*n + 0.2*sin(2*pi*0.002*n)) + ...
    0.3*sin(2*pi*0.055*n + 0.4);

% Unknown nonlinear mapping y(n) = phi{x(n)} + noise
noise_power = 0.05;
noise_std = sqrt(noise_power);

y_clean = 0.8*tanh(1.5*x) + 0.25*x.^2 - 0.08*x.^3 + 0.15*sin(2*x);
y = y_clean + noise_std*randn(size(y_clean));

%% ------------------------------------------------------------
% 2. Build one-step prediction dataset
%    predict y(n) from past samples [y(n-1), ..., y(n-4)]
%% ------------------------------------------------------------
order = 4;
num_samples = N - order;

X = zeros(order, num_samples);
T = zeros(1, num_samples);

for k = 1:num_samples
    idx = k + order;
    X(:,k) = y(idx-1:-1:idx-order);
    T(k) = y(idx);
end

% Train/test split
train_ratio = 0.7;
num_train = floor(train_ratio * num_samples);

X_train = X(:,1:num_train);
T_train = T(:,1:num_train);

X_test  = X(:,num_train+1:end);
T_test  = T(:,num_train+1:end);

%% ------------------------------------------------------------
% 3. Normalise inputs using training statistics
%% ------------------------------------------------------------
mu_X = mean(X_train, 2);
std_X = std(X_train, 0, 2) + 1e-8;

X_train_n = (X_train - mu_X) ./ std_X;
X_test_n  = (X_test  - mu_X) ./ std_X;

%% ------------------------------------------------------------
% 4. Training settings
%% ------------------------------------------------------------
epochs = 20000;
lr = 0.01;

% Containers for test MSE vs epoch
mse_test_perc = zeros(epochs,1);
mse_test_dnn  = zeros(epochs,1);

%% ------------------------------------------------------------
% 5. Simple dynamical perceptron
%    yhat = tanh(w*x + b)
%% ------------------------------------------------------------
W1 = 0.1*randn(1, order);
b1 = 0;

for ep = 1:epochs
    % Forward pass (train)
    Z_train = W1 * X_train_n + b1;
    Y_train = tanh(Z_train);

    E_train = Y_train - T_train;
    dZ = 2 * E_train .* (1 - Y_train.^2) / num_train;

    % Gradients
    dW1 = dZ * X_train_n';
    db1 = sum(dZ, 2);

    % Update
    W1 = W1 - lr * dW1;
    b1 = b1 - lr * db1;

    % Test MSE
    Y_test = tanh(W1 * X_test_n + b1);
    mse_test_perc(ep) = mean((Y_test - T_test).^2);
end

%% ------------------------------------------------------------
% 6. Deep neural network with 4 hidden layers
%    ReLU hidden units + linear output
%% ------------------------------------------------------------
h1 = 32;
h2 = 32;
h3 = 16;
h4 = 16;

W1d = 0.05*randn(h1, order);   b1d = zeros(h1,1);
W2d = 0.05*randn(h2, h1);      b2d = zeros(h2,1);
W3d = 0.05*randn(h3, h2);      b3d = zeros(h3,1);
W4d = 0.05*randn(h4, h3);      b4d = zeros(h4,1);
W5d = 0.05*randn(1,  h4);      b5d = 0;

relu  = @(z) max(0,z);
drelu = @(z) double(z > 0);

for ep = 1:epochs
    %% Forward pass (train)
    Z1 = W1d * X_train_n + b1d;
    A1 = relu(Z1);

    Z2 = W2d * A1 + b2d;
    A2 = relu(Z2);

    Z3 = W3d * A2 + b3d;
    A3 = relu(Z3);

    Z4 = W4d * A3 + b4d;
    A4 = relu(Z4);

    Z5 = W5d * A4 + b5d;   % linear output
    Y_train = Z5;

    %% Backpropagation
    E = Y_train - T_train;
    dZ5 = 2 * E / num_train;

    dW5 = dZ5 * A4';
    db5 = sum(dZ5, 2);

    dA4 = W5d' * dZ5;
    dZ4 = dA4 .* drelu(Z4);
    dW4 = dZ4 * A3';
    db4 = sum(dZ4, 2);

    dA3 = W4d' * dZ4;
    dZ3 = dA3 .* drelu(Z3);
    dW3 = dZ3 * A2';
    db3 = sum(dZ3, 2);

    dA2 = W3d' * dZ3;
    dZ2 = dA2 .* drelu(Z2);
    dW2 = dZ2 * A1';
    db2 = sum(dZ2, 2);

    dA1 = W2d' * dZ2;
    dZ1 = dA1 .* drelu(Z1);
    dW1 = dZ1 * X_train_n';
    db1 = sum(dZ1, 2);

    %% Update
    W5d = W5d - lr * dW5;   b5d = b5d - lr * db5;
    W4d = W4d - lr * dW4;   b4d = b4d - lr * db4;
    W3d = W3d - lr * dW3;   b3d = b3d - lr * db3;
    W2d = W2d - lr * dW2;   b2d = b2d - lr * db2;
    W1d = W1d - lr * dW1;   b1d = b1d - lr * db1;

    %% Test MSE
    Z1t = W1d * X_test_n + b1d;  A1t = relu(Z1t);
    Z2t = W2d * A1t + b2d;       A2t = relu(Z2t);
    Z3t = W3d * A2t + b3d;       A3t = relu(Z3t);
    Z4t = W4d * A3t + b4d;       A4t = relu(Z4t);
    Yt  = W5d * A4t + b5d;

    mse_test_dnn(ep) = mean((Yt - T_test).^2);
end

%% ------------------------------------------------------------
% 7. Final predictions for visual comparison
%% ------------------------------------------------------------
% Final perceptron prediction
Y_test_perc = tanh(W1 * X_test_n + b1);

% Final DNN prediction
Z1t = W1d * X_test_n + b1d;  A1t = relu(Z1t);
Z2t = W2d * A1t + b2d;       A2t = relu(Z2t);
Z3t = W3d * A2t + b3d;       A3t = relu(Z3t);
Z4t = W4d * A3t + b4d;       A4t = relu(Z4t);
Y_test_dnn = W5d * A4t + b5d;

%% ------------------------------------------------------------
% 8. Find minimum test MSE and epochs
%% ------------------------------------------------------------
[min_mse_perc, idx_perc] = min(mse_test_perc);
[min_mse_dnn,  idx_dnn]  = min(mse_test_dnn);

fprintf('Simple dynamical perceptron:\n');
fprintf('  Minimum test MSE = %.6f at epoch %d\n', min_mse_perc, idx_perc);

fprintf('Deep neural network:\n');
fprintf('  Minimum test MSE = %.6f at epoch %d\n', min_mse_dnn, idx_dnn);

%% ------------------------------------------------------------
% 9. Plots
%% ------------------------------------------------------------
figure('Color','w','Position',[100 100 1200 450]);

subplot(1,2,1)
plot(T_test, 'k', 'LineWidth', 1.2); hold on;
plot(Y_test_perc, 'Color', [0.850 0.325 0.098], 'LineWidth', 1.2);
plot(Y_test_dnn,  'Color', [0.466 0.674 0.188], 'LineWidth', 1.2);
grid on;
xlabel('Test sample index');
ylabel('Amplitude');
title('True signal and model predictions');
legend('True signal', 'Dynamical perceptron', '4-hidden-layer DNN', 'Location', 'best');

subplot(1,2,2)
plot(mse_test_perc, 'Color', [0.850 0.325 0.098], 'LineWidth', 1.3); hold on;
plot(mse_test_dnn,  'Color', [0.466 0.674 0.188], 'LineWidth', 1.3);
plot(idx_perc, min_mse_perc, 'o', 'Color', [0.850 0.325 0.098], 'MarkerFaceColor', [0.850 0.325 0.098]);
plot(idx_dnn,  min_mse_dnn,  'o', 'Color', [0.466 0.674 0.188], 'MarkerFaceColor', [0.466 0.674 0.188]);
grid on;
xlabel('Epoch');
ylabel('Test MSE');
title('Test performance against epoch number');
legend('Dynamical perceptron', '4-hidden-layer DNN', 'Location', 'best');

%% ------------------------------------------------------------
% 10. Optional: training curves can also be stored similarly
% if your marker wants both training and testing performance.
%% ------------------------------------------------------------

%% Q4.7 - Effect of Noise Power on Deep Learning Prediction
% Repeats the nonlinear prediction experiment for different noise powers
% and compares a simple dynamical perceptron with a 4-hidden-layer DNN.

clear; close all; clc;
rng(1);

%% ------------------------------------------------------------
% 1. Common settings
%% ------------------------------------------------------------
N = 2000;
n = (1:N)';

order = 4;              % one-step prediction from 4 past samples
epochs = 20000;
lr = 0.01;

noise_powers = [0.01 0.03 0.05 0.1 0.2];

train_ratio = 0.7;

% Perceptron and DNN hidden sizes
h1 = 32;
h2 = 32;
h3 = 16;
h4 = 16;

% Store results
num_noise = length(noise_powers);
best_mse_perc = zeros(num_noise,1);
best_mse_dnn  = zeros(num_noise,1);
best_ep_perc  = zeros(num_noise,1);
best_ep_dnn   = zeros(num_noise,1);

% To store one representative set of curves for each noise power
mse_curves_perc = zeros(epochs, num_noise);
mse_curves_dnn  = zeros(epochs, num_noise);

%% ------------------------------------------------------------
% 2. Base nonlinear clean signal
%% ------------------------------------------------------------
x = 1.2*sin(2*pi*0.003*n) + ...
    0.9*cos(2*pi*0.011*n + 0.6) + ...
    0.6*sin(2*pi*0.021*n.^1.02) + ...
    0.5*cos(2*pi*0.035*n + 0.2*sin(2*pi*0.002*n)) + ...
    0.3*sin(2*pi*0.055*n + 0.4);

y_clean = 0.8*tanh(1.5*x) + 0.25*x.^2 - 0.08*x.^3 + 0.15*sin(2*x);

%% ------------------------------------------------------------
% 3. Loop over different noise powers
%% ------------------------------------------------------------
for np_idx = 1:num_noise

    noise_power = noise_powers(np_idx);
    noise_std = sqrt(noise_power);

    % Noisy target
    y = y_clean + noise_std*randn(size(y_clean));

    %% Build one-step prediction dataset
    num_samples = N - order;
    X = zeros(order, num_samples);
    T = zeros(1, num_samples);

    for k = 1:num_samples
        idx = k + order;
        X(:,k) = y(idx-1:-1:idx-order);
        T(k) = y(idx);
    end

    % Train/test split
    num_train = floor(train_ratio * num_samples);

    X_train = X(:,1:num_train);
    T_train = T(:,1:num_train);

    X_test = X(:,num_train+1:end);
    T_test = T(:,num_train+1:end);

    % Normalise inputs using training stats
    mu_X = mean(X_train, 2);
    std_X = std(X_train, 0, 2) + 1e-8;

    X_train_n = (X_train - mu_X) ./ std_X;
    X_test_n  = (X_test  - mu_X) ./ std_X;

    %% --------------------------------------------------------
    % 3a. Simple dynamical perceptron
    %     yhat = tanh(w*x + b)
    %% --------------------------------------------------------
    Wp = 0.1*randn(1, order);
    bp = 0;

    mse_test_perc = zeros(epochs,1);

    for ep = 1:epochs
        % Forward
        Zp = Wp * X_train_n + bp;
        Yp = tanh(Zp);

        % Loss gradient
        Ep = Yp - T_train;
        dZp = 2 * Ep .* (1 - Yp.^2) / num_train;

        dWp = dZp * X_train_n';
        dbp = sum(dZp, 2);

        % Update
        Wp = Wp - lr * dWp;
        bp = bp - lr * dbp;

        % Test performance
        Yp_test = tanh(Wp * X_test_n + bp);
        mse_test_perc(ep) = mean((Yp_test - T_test).^2);
    end

    %% --------------------------------------------------------
    % 3b. Deep neural network with 4 hidden layers
    %% --------------------------------------------------------
    W1 = 0.05*randn(h1, order);   b1 = zeros(h1,1);
    W2 = 0.05*randn(h2, h1);      b2 = zeros(h2,1);
    W3 = 0.05*randn(h3, h2);      b3 = zeros(h3,1);
    W4 = 0.05*randn(h4, h3);      b4 = zeros(h4,1);
    W5 = 0.05*randn(1,  h4);      b5 = 0;

    relu  = @(z) max(0,z);
    drelu = @(z) double(z > 0);

    mse_test_dnn = zeros(epochs,1);

    for ep = 1:epochs
        % Forward pass
        Z1 = W1 * X_train_n + b1;  A1 = relu(Z1);
        Z2 = W2 * A1 + b2;         A2 = relu(Z2);
        Z3 = W3 * A2 + b3;         A3 = relu(Z3);
        Z4 = W4 * A3 + b4;         A4 = relu(Z4);
        Z5 = W5 * A4 + b5;         Yd = Z5;   % linear output

        % Backprop
        E = Yd - T_train;
        dZ5 = 2 * E / num_train;

        dW5 = dZ5 * A4';
        db5 = sum(dZ5,2);

        dA4 = W5' * dZ5;
        dZ4 = dA4 .* drelu(Z4);
        dW4 = dZ4 * A3';
        db4 = sum(dZ4,2);

        dA3 = W4' * dZ4;
        dZ3 = dA3 .* drelu(Z3);
        dW3 = dZ3 * A2';
        db3 = sum(dZ3,2);

        dA2 = W3' * dZ3;
        dZ2 = dA2 .* drelu(Z2);
        dW2 = dZ2 * A1';
        db2 = sum(dZ2,2);

        dA1 = W2' * dZ2;
        dZ1 = dA1 .* drelu(Z1);
        dW1 = dZ1 * X_train_n';
        db1 = sum(dZ1,2);

        % Update
        W5 = W5 - lr * dW5;   b5 = b5 - lr * db5;
        W4 = W4 - lr * dW4;   b4 = b4 - lr * db4;
        W3 = W3 - lr * dW3;   b3 = b3 - lr * db3;
        W2 = W2 - lr * dW2;   b2 = b2 - lr * db2;
        W1 = W1 - lr * dW1;   b1 = b1 - lr * db1;

        % Test performance
        Z1t = W1 * X_test_n + b1;  A1t = relu(Z1t);
        Z2t = W2 * A1t + b2;       A2t = relu(Z2t);
        Z3t = W3 * A2t + b3;       A3t = relu(Z3t);
        Z4t = W4 * A3t + b4;       A4t = relu(Z4t);
        Yt  = W5 * A4t + b5;

        mse_test_dnn(ep) = mean((Yt - T_test).^2);
    end

    %% Store results
    mse_curves_perc(:,np_idx) = mse_test_perc;
    mse_curves_dnn(:,np_idx)  = mse_test_dnn;

    [best_mse_perc(np_idx), best_ep_perc(np_idx)] = min(mse_test_perc);
    [best_mse_dnn(np_idx),  best_ep_dnn(np_idx)]  = min(mse_test_dnn);

    fprintf('\nNoise power = %.3f\n', noise_power);
    fprintf('  Perceptron: best test MSE = %.6f at epoch %d\n', ...
        best_mse_perc(np_idx), best_ep_perc(np_idx));
    fprintf('  DNN       : best test MSE = %.6f at epoch %d\n', ...
        best_mse_dnn(np_idx), best_ep_dnn(np_idx));
end

%% ------------------------------------------------------------
% 4. Plot test MSE vs epoch for each noise power
%% ------------------------------------------------------------
figure('Color','w','Position',[80 80 1300 800]);

for np_idx = 1:num_noise
    subplot(2,3,np_idx);

    plot(mse_curves_perc(:,np_idx), 'Color', [0.850 0.325 0.098], 'LineWidth', 1.2); hold on;
    plot(mse_curves_dnn(:,np_idx),  'Color', [0.466 0.674 0.188], 'LineWidth', 1.2);

    plot(best_ep_perc(np_idx), best_mse_perc(np_idx), 'o', ...
        'Color', [0.850 0.325 0.098], 'MarkerFaceColor', [0.850 0.325 0.098]);

    plot(best_ep_dnn(np_idx), best_mse_dnn(np_idx), 'o', ...
        'Color', [0.466 0.674 0.188], 'MarkerFaceColor', [0.466 0.674 0.188]);

    grid on;
    xlabel('Epoch');
    ylabel('Test MSE');
    title(['Noise power = ', num2str(noise_powers(np_idx))]);

    if np_idx == 1
        legend('Dynamical perceptron', '4-hidden-layer DNN', 'Location', 'best');
    end
end

sgtitle('Effect of noise power on test performance');

%% ------------------------------------------------------------
% 5. Summary plot: best test MSE vs noise power
%% ------------------------------------------------------------
figure('Color','w','Position',[100 100 1100 400]);

subplot(1,2,1)
plot(noise_powers, best_mse_perc, '-o', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.5, 'MarkerFaceColor', [0.850 0.325 0.098]); hold on;
plot(noise_powers, best_mse_dnn, '-o', 'Color', [0.466 0.674 0.188], ...
    'LineWidth', 1.5, 'MarkerFaceColor', [0.466 0.674 0.188]);
grid on;
xlabel('Noise power');
ylabel('Best test MSE');
title('Best test MSE vs noise power');
legend('Dynamical perceptron', '4-hidden-layer DNN', 'Location', 'best');

subplot(1,2,2)
plot(noise_powers, best_ep_perc, '-o', 'Color', [0.850 0.325 0.098], ...
    'LineWidth', 1.5, 'MarkerFaceColor', [0.850 0.325 0.098]); hold on;
plot(noise_powers, best_ep_dnn, '-o', 'Color', [0.466 0.674 0.188], ...
    'LineWidth', 1.5, 'MarkerFaceColor', [0.466 0.674 0.188]);
grid on;
xlabel('Noise power');
ylabel('Epoch of best test MSE');
title('Best epoch vs noise power');
legend('Dynamical perceptron', '4-hidden-layer DNN', 'Location', 'best');