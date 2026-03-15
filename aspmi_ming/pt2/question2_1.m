%% Assignment 2.1 - LMS and Leaky LMS for AR(2) Identification
close all; clear; clc;

%% ========================================================================
%% Parameters
%% ========================================================================
N         = 1000;      % Number of samples
Num_iter  = 100;       % Number of Monte Carlo trials
noise_var = 0.25;      % Driving noise variance
a_true    = [1 -0.1 -0.8];   % AR polynomial: x(n)-0.1x(n-1)-0.8x(n-2)=eta(n)
b_true    = 1;
num_coef  = length(a_true)-1;

% LMS step sizes
mu1 = 0.05;
mu2 = 0.01;

% Steady-state indices
t_steady_05 = 500;
t_steady_01 = 700;

% Theoretical correlation matrix from part (a)
R = [25/27 25/54; 
     25/54 25/27];

% Plot colours
c_blue   = [0.00 0.45 0.74];
c_orange = [0.85 0.33 0.10];
c_green  = [0.20 0.60 0.30];
c_purple = [0.49 0.18 0.56];
c_red    = [0.80 0.20 0.20];
c_grey   = [0.35 0.35 0.35];

lw_main  = 1.6;
lw_ref   = 1.4;

rng('default');

%% ========================================================================
%% Part b) One realisation
%% ========================================================================
h = sqrt(noise_var) * randn(N,1);
x = filter(b_true, a_true, h);

[x_hat1, error1, weights1] = adaptive_lms(x, mu1, num_coef);
[x_hat2, error2, weights2] = adaptive_lms(x, mu2, num_coef);

figure('Name','Part b - LMS Error Curves','Color','w','Position',[100 100 1200 450]);

tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(pow2db(error1.^2 + eps), 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(pow2db(error2.^2 + eps), 'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Sample','FontSize',11);
ylabel('Squared error (dB)','FontSize',11);
legend('\mu = 0.05','\mu = 0.01','Location','best');
title('Single realisation','FontSize',12);

%% ========================================================================
%% Part b) 100 realisations
%% ========================================================================
error_05 = zeros(Num_iter,N);
error_01 = zeros(Num_iter,N);

coefficients_05_1 = zeros(Num_iter,N);
coefficients_05_2 = zeros(Num_iter,N);
coefficients_01_1 = zeros(Num_iter,N);
coefficients_01_2 = zeros(Num_iter,N);

for iter = 1:Num_iter
    h = sqrt(noise_var) * randn(N,1);
    x = filter(b_true, a_true, h);

    [~, error1, weights1] = adaptive_lms(x, mu1, num_coef);
    [~, error2, weights2] = adaptive_lms(x, mu2, num_coef);

    error_05(iter,:) = error1(:).';
    error_01(iter,:) = error2(:).';

    coefficients_05_1(iter,:) = weights1(1,:);
    coefficients_05_2(iter,:) = weights1(2,:);
    coefficients_01_1(iter,:) = weights2(1,:);
    coefficients_01_2(iter,:) = weights2(2,:);
end

% Average learning curves in dB domain
error_05_mean_dB = mean(pow2db(error_05.^2 + eps), 1);
error_01_mean_dB = mean(pow2db(error_01.^2 + eps), 1);

nexttile;
plot(error_05_mean_dB, 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(error_01_mean_dB, 'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Sample','FontSize',11);
ylabel('Squared error (dB)','FontSize',11);
legend('\mu = 0.05','\mu = 0.01','Location','best');
title('Ensemble-averaged learning curve (100 trials)','FontSize',12);

%% ========================================================================
%% Part c) Misadjustment
%% ========================================================================
MSE_05 = mean(error_05(:, t_steady_05:end).^2, 'all');
MSE_01 = mean(error_01(:, t_steady_01:end).^2, 'all');

EMSE_05 = MSE_05 - noise_var;
EMSE_01 = MSE_01 - noise_var;

M_05 = EMSE_05 / noise_var;
M_01 = EMSE_01 / noise_var;

% Theoretical approximation: M ≈ (mu/2) Tr(R)
M_05_true = (mu1/2) * trace(R);
M_01_true = (mu2/2) * trace(R);

fprintf('\n================ Part (c): Misadjustment ================\n');
fprintf('mu = %.2f | EMSE = %.4f | M_est = %.4f | M_theory = %.4f\n', mu1, EMSE_05, M_05, M_05_true);
fprintf('mu = %.2f | EMSE = %.4f | M_est = %.4f | M_theory = %.4f\n', mu2, EMSE_01, M_01, M_01_true);

%% ========================================================================
%% Part d) Coefficient trajectories, bias, variance, MSE
%% ========================================================================
a_pred_05_1 = mean(coefficients_05_1,1);
a_pred_05_2 = mean(coefficients_05_2,1);

a_pred_01_1 = mean(coefficients_01_1,1);
a_pred_01_2 = mean(coefficients_01_2,1);

a1_true = -a_true(2);   % 0.1
a2_true = -a_true(3);   % 0.8

% Steady-state means
a_pred_05_1_mean = mean(a_pred_05_1(t_steady_05:end));
a_pred_05_2_mean = mean(a_pred_05_2(t_steady_05:end));
a_pred_01_1_mean = mean(a_pred_01_1(t_steady_01:end));
a_pred_01_2_mean = mean(a_pred_01_2(t_steady_01:end));

fprintf('\n================ Part (d): Steady-state coefficient estimates ================\n');
fprintf('mu = %.2f | a1_hat = %.4f | a2_hat = %.4f\n', mu1, a_pred_05_1_mean, a_pred_05_2_mean);
fprintf('mu = %.2f | a1_hat = %.4f | a2_hat = %.4f\n', mu2, a_pred_01_1_mean, a_pred_01_2_mean);

% Variance across ensemble at each time instant
a_pred_05_1_var = var(coefficients_05_1,0,1);
a_pred_05_2_var = var(coefficients_05_2,0,1);
a_pred_01_1_var = var(coefficients_01_1,0,1);
a_pred_01_2_var = var(coefficients_01_2,0,1);

% Bias across ensemble at each time instant
a_pred_05_1_bias = mean(coefficients_05_1,1) - a1_true;
a_pred_05_2_bias = mean(coefficients_05_2,1) - a2_true;
a_pred_01_1_bias = mean(coefficients_01_1,1) - a1_true;
a_pred_01_2_bias = mean(coefficients_01_2,1) - a2_true;

% MSE = bias^2 + variance
a_pred_05_1_mse = a_pred_05_1_bias.^2 + a_pred_05_1_var;
a_pred_05_2_mse = a_pred_05_2_bias.^2 + a_pred_05_2_var;
a_pred_01_1_mse = a_pred_01_1_bias.^2 + a_pred_01_1_var;
a_pred_01_2_mse = a_pred_01_2_bias.^2 + a_pred_01_2_var;

%% Plot coefficient evolution
figure('Name','Part d - Coefficient Evolution','Color','w','Position',[100 100 1200 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(a_pred_05_1,'Color',c_blue,'LineWidth',lw_main); hold on;
plot(a_pred_05_2,'Color',c_orange,'LineWidth',lw_main);
yline(a1_true,'--','Color',c_green,'LineWidth',lw_ref);
yline(a2_true,'--','Color',c_grey,'LineWidth',lw_ref);
grid on; box on;
xlabel('Sample'); ylabel('Coefficient value');
ylim([0 1]);
title('\mu = 0.05','FontSize',12);
legend('\hat{a}_1','\hat{a}_2','a_1 true','a_2 true','Location','best');

nexttile;
plot(a_pred_01_1,'Color',c_blue,'LineWidth',lw_main); hold on;
plot(a_pred_01_2,'Color',c_orange,'LineWidth',lw_main);
yline(a1_true,'--','Color',c_green,'LineWidth',lw_ref);
yline(a2_true,'--','Color',c_grey,'LineWidth',lw_ref);
grid on; box on;
xlabel('Sample'); ylabel('Coefficient value');
ylim([0 1]);
title('\mu = 0.01','FontSize',12);
legend('\hat{a}_1','\hat{a}_2','a_1 true','a_2 true','Location','best');

%% Plot variance and bias
figure('Name','Part d - Bias and Variance','Color','w','Position',[80 80 1200 700]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(a_pred_05_1_var,'Color',c_blue,'LineWidth',lw_main); hold on;
plot(a_pred_01_1_var,'Color',c_orange,'LineWidth',lw_main);
grid on; box on;
xlabel('Sample'); ylabel('Variance');
title('Variance of \hat{a}_1');
legend('\mu = 0.05','\mu = 0.01','Location','best');

nexttile;
plot(a_pred_05_2_var,'Color',c_blue,'LineWidth',lw_main); hold on;
plot(a_pred_01_2_var,'Color',c_orange,'LineWidth',lw_main);
grid on; box on;
xlabel('Sample'); ylabel('Variance');
title('Variance of \hat{a}_2');
legend('\mu = 0.05','\mu = 0.01','Location','best');

nexttile;
plot(abs(a_pred_05_1_bias),'Color',c_blue,'LineWidth',lw_main); hold on;
plot(abs(a_pred_01_1_bias),'Color',c_orange,'LineWidth',lw_main);
grid on; box on;
xlabel('Sample'); ylabel('|Bias|');
title('Bias magnitude of \hat{a}_1');
legend('\mu = 0.05','\mu = 0.01','Location','best');

nexttile;
plot(abs(a_pred_05_2_bias),'Color',c_blue,'LineWidth',lw_main); hold on;
plot(abs(a_pred_01_2_bias),'Color',c_orange,'LineWidth',lw_main);
grid on; box on;
xlabel('Sample'); ylabel('|Bias|');
title('Bias magnitude of \hat{a}_2');
legend('\mu = 0.05','\mu = 0.01','Location','best');

%% Plot coefficient MSE
figure('Name','Part d - Coefficient MSE','Color','w','Position',[100 100 1200 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(a_pred_05_1_mse,'Color',c_blue,'LineWidth',lw_main); hold on;
plot(a_pred_01_1_mse,'Color',c_orange,'LineWidth',lw_main);
grid on; box on;
xlabel('Sample'); ylabel('MSE');
title('MSE of \hat{a}_1');
legend('\mu = 0.05','\mu = 0.01','Location','best');

nexttile;
plot(a_pred_05_2_mse,'Color',c_blue,'LineWidth',lw_main); hold on;
plot(a_pred_01_2_mse,'Color',c_orange,'LineWidth',lw_main);
grid on; box on;
xlabel('Sample'); ylabel('MSE');
title('MSE of \hat{a}_2');
legend('\mu = 0.05','\mu = 0.01','Location','best');

%% ========================================================================
%% Part f) Leaky LMS
%% ========================================================================
% Try one mu and several gamma values to show shrinkage / bias effect
mu_leaky = 0.05;
gamma_list = [0.2 0.5 0.8];
t_steady_leaky = 600;

coeff_leaky_a1 = zeros(length(gamma_list), Num_iter, N);
coeff_leaky_a2 = zeros(length(gamma_list), Num_iter, N);

for g_idx = 1:length(gamma_list)
    g = gamma_list(g_idx);

    for iter = 1:Num_iter
        h = sqrt(noise_var) * randn(N,1);
        x = filter(b_true, a_true, h);

        [~,~,weights] = leaky_lms(x, mu_leaky, g, num_coef);

        coeff_leaky_a1(g_idx,iter,:) = weights(1,:);
        coeff_leaky_a2(g_idx,iter,:) = weights(2,:);
    end
end

% Ensemble averages
a1_leaky_mean = squeeze(mean(coeff_leaky_a1,2));
a2_leaky_mean = squeeze(mean(coeff_leaky_a2,2));

% Steady-state means
fprintf('\n================ Part (f): Leaky LMS steady-state estimates ================\n');
for g_idx = 1:length(gamma_list)
    g = gamma_list(g_idx);

    a1_ss = mean(a1_leaky_mean(g_idx,t_steady_leaky:end));
    a2_ss = mean(a2_leaky_mean(g_idx,t_steady_leaky:end));

    abs_err_1 = abs(a1_ss-a1_true)/a1_true*100;
    abs_err_2 = abs(a2_ss-a2_true)/a2_true*100;

    fprintf('gamma = %.2f | a1_hat = %.4f | a2_hat = %.4f | err_a1 = %.2f%% | err_a2 = %.2f%%\n', ...
        g, a1_ss, a2_ss, abs_err_1, abs_err_2);
end

%% Plot leaky coefficient evolution
figure('Name','Part f - Leaky LMS Coefficient Evolution','Color','w','Position',[60 60 1300 700]);
tiledlayout(2, length(gamma_list), 'Padding','compact','TileSpacing','compact');

for g_idx = 1:length(gamma_list)
    g = gamma_list(g_idx);

    nexttile;
    plot(a1_leaky_mean(g_idx,:), 'Color', c_blue,   'LineWidth', lw_main); hold on;
    yline(a1_true, '--', 'Color', c_green, 'LineWidth', lw_ref);
    grid on; box on;
    xlabel('Sample'); ylabel('\hat{a}_1');
    title(['Leaky LMS: \gamma = ', num2str(g)]);
    legend('\hat{a}_1','a_1 true','Location','best');

    nexttile;
    plot(a2_leaky_mean(g_idx,:), 'Color', c_orange, 'LineWidth', lw_main); hold on;
    yline(a2_true, '--', 'Color', c_grey, 'LineWidth', lw_ref);
    grid on; box on;
    xlabel('Sample'); ylabel('\hat{a}_2');
    title(['Leaky LMS: \gamma = ', num2str(g)]);
    legend('\hat{a}_2','a_2 true','Location','best');
end

%% Compare original signal and prediction using one leaky setting
g_demo = gamma_list(2);  % choose gamma = 0.5 for display
idx_demo = find(gamma_list == g_demo, 1);

a1_demo = mean(a1_leaky_mean(idx_demo,t_steady_leaky:end));
a2_demo = mean(a2_leaky_mean(idx_demo,t_steady_leaky:end));

h = sqrt(noise_var) * randn(N,1);
x = filter(b_true, a_true, h);

x_hat_leaky = zeros(N,1);
for n = 3:N
    x_hat_leaky(n) = a1_demo*x(n-1) + a2_demo*x(n-2);
end

figure('Name','Part f - Signal Reconstruction with Leaky LMS','Color','w','Position',[120 120 1100 400]);
plot(x, 'Color', c_blue, 'LineWidth', 1.2); hold on;
plot(x_hat_leaky, 'Color', c_red, 'LineWidth', 1.2);
grid on; box on;
xlabel('Sample'); ylabel('Amplitude');
title(['Signal and one-step prediction using leaky LMS (\gamma = ', num2str(g_demo), ')']);
legend('True x(n)','Predicted \hat{x}(n)','Location','best');

%% Estimate regularised optimal solution: w_opt = (R + gamma I)^(-1) p
Num_iter_opt = 1000;
p_steady_mtx = zeros(2, Num_iter_opt);

for iter = 1:Num_iter_opt
    h = sqrt(noise_var) * randn(N,1);
    x = filter(b_true, a_true, h);

    p = zeros(2, N-2);
    for n = 3:N
        x_vec = [x(n-1); x(n-2)];
        p(:,n-2) = x(n) * x_vec;
    end
    p_steady_mtx(:,iter) = mean(p(:, t_steady_leaky-2:end), 2);
end

p_steady_mean = mean(p_steady_mtx, 2);

fprintf('\nRegularised theoretical solutions:\n');
for g_idx = 1:length(gamma_list)
    g = gamma_list(g_idx);
    wopt = (R + g*eye(2)) \ p_steady_mean;
    fprintf('gamma = %.2f | w_opt = [%.4f, %.4f]^T\n', g, wopt(1), wopt(2));
end

%% Leaky LMS bias / variance / MSE for each gamma
figure('Name','Part f - Leaky LMS MSE','Color','w','Position',[80 80 1300 400]);
tiledlayout(1, length(gamma_list), 'Padding','compact','TileSpacing','compact');

for g_idx = 1:length(gamma_list)
    bias_a1 = squeeze(mean(coeff_leaky_a1(g_idx,:,:),2)).' - a1_true;
    var_a1  = squeeze(var(coeff_leaky_a1(g_idx,:,:),0,2)).';
    mse_a1  = bias_a1.^2 + var_a1;

    bias_a2 = squeeze(mean(coeff_leaky_a2(g_idx,:,:),2)).' - a2_true;
    var_a2  = squeeze(var(coeff_leaky_a2(g_idx,:,:),0,2)).';
    mse_a2  = bias_a2.^2 + var_a2;

    nexttile;
    plot(mse_a1, 'Color', c_blue,   'LineWidth', lw_main); hold on;
    plot(mse_a2, 'Color', c_orange, 'LineWidth', lw_main);
    grid on; box on;
    xlabel('Sample'); ylabel('MSE');
    title(['Leaky LMS MSE, \gamma = ', num2str(gamma_list(g_idx))]);
    legend('MSE of \hat{a}_1','MSE of \hat{a}_2','Location','best');
end