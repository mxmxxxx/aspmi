%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        Task 1.6.1  Partial Least Squares (PLS)      %%
%%      Compare PLS vs PCR on test set (r = 1..10)    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear;

%% -------------------------
% Load data
%% -------------------------
load('PCAPCR.mat');   % expects X, Y, Xtest, Ytest

%% -------------------------
% Add noise to training input (same as earlier parts)
%% -------------------------
Nx     = sqrt(4) * randn(size(X));
Xnoise = X + Nx;

%% -------------------------
% Mean-centre (important!)
%% -------------------------
Xn = Xnoise - mean(Xnoise,1);
Yn = Y      - mean(Y,1);

Xt = Xtest  - mean(Xnoise,1);   % centre test using TRAIN mean
Yt = Ytest  - mean(Y,1);

%% -------------------------
% PCR + PLS over r = 1..10
%% -------------------------
maxR = min(10, size(Xn,2));
mse_pcr = zeros(maxR,1);
mse_pls = zeros(maxR,1);

for r = 1:maxR

    % =======================
    % PCR
    % =======================
    [U,S,V] = svd(Xn,'econ');

    Ur = U(:,1:r);
    Sr = S(1:r,1:r);
    Vr = V(:,1:r);

    B_pcr = Vr * (Sr \ (Ur' * Yn));
    Yhat_pcr = Xt * B_pcr;

    err_pcr = Yt - Yhat_pcr;
    mse_pcr(r) = mean(err_pcr(:).^2);

    % =======================
    % PLS (default NIPALS)
    % =======================
    % NOTE: no 'Algorithm' argument (older MATLAB compatible)
    [~,~,~,~,beta_pls] = plsregress(Xn, Yn, r);

    % beta includes intercept
    Yhat_pls = [ones(size(Xt,1),1) Xt] * beta_pls;

    err_pls = Yt - Yhat_pls;
    mse_pls(r) = mean(err_pls(:).^2);
end

%% -------------------------
% Plot results
%% -------------------------
figure('Color','w');
plot(1:maxR, mse_pcr, '-o', 'LineWidth', 1.5); hold on;
plot(1:maxR, mse_pls, '-s', 'LineWidth', 1.5);
grid on;
xlabel('Number of components r');
ylabel('Test MSE');
title('PCR vs PLS Test-Set Performance');
legend('PCR','PLS','Location','best');

%% -------------------------
% Report best models
%% -------------------------
[bestPCR, rPCR] = min(mse_pcr);
[bestPLS, rPLS] = min(mse_pls);

fprintf('Best PCR: r = %d, Test MSE = %.6f\n', rPCR, bestPCR);
fprintf('Best PLS: r = %d, Test MSE = %.6f\n', rPLS, bestPLS);
