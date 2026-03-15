%% Assignment 2.2 - Adaptive Step Sizes
close all; clear; clc;

%% ========================================================================
%% Global plotting style
%% ========================================================================
c_blue   = [0.00 0.45 0.74];
c_orange = [0.85 0.33 0.10];
c_green  = [0.20 0.60 0.30];
c_purple = [0.49 0.18 0.56];
c_red    = [0.80 0.20 0.20];
c_grey   = [0.35 0.35 0.35];
c_yellow = [0.93 0.69 0.13];

lw_main = 1.5;
lw_ref  = 1.2;

rng('default');

%% ========================================================================
%% Part (a)
%% ========================================================================
Num_iter  = 100;     % Number of Monte Carlo trials
N         = 2000;    % Number of samples
a         = 1;
b         = [1 0.9]; % MA(1): x(n) = h(n) + 0.9 h(n-1)
num_coef  = length(b);
noise_var = 0.5;

% Parameters
mu_fixed = [0.01, 0.1];
mu_0     = 0.1;      % Initial step size for GASS methods
rho      = 0.005;
alpha    = 0.9;

% Preallocate error matrices: N x Num_iter
error_lms_001   = zeros(N, Num_iter);
error_lms_01    = zeros(N, Num_iter);
error_benv      = zeros(N, Num_iter);
error_ang       = zeros(N, Num_iter);
error_mx        = zeros(N, Num_iter);

% Preallocate weight trajectories: Num_iter x N
w_lms_001 = zeros(Num_iter, N);
w_lms_01  = zeros(Num_iter, N);
w_benv    = zeros(Num_iter, N);
w_ang     = zeros(Num_iter, N);
w_mx      = zeros(Num_iter, N);

for iter = 1:Num_iter
    % Generate white Gaussian noise
    h = sqrt(noise_var) * randn(N,1);

    % Generate MA(1) signal
    x = filter(b, a, h);

    % 1) Standard LMS, mu = 0.01
    [~, e1, w1] = adaptive_lms_MA(h, x, mu_fixed(1), num_coef);
    error_lms_001(:,iter) = e1(:);
    w_lms_001(iter,:)     = w1(1,:);

    % 2) Standard LMS, mu = 0.1
    [~, e2, w2] = adaptive_lms_MA(h, x, mu_fixed(2), num_coef);
    error_lms_01(:,iter) = e2(:);
    w_lms_01(iter,:)     = w2(1,:);

    % 3) Benveniste
    [~, ~, e3, w3] = benveniste(h, x, mu_0, rho, num_coef);
    error_benv(:,iter) = e3(:);
    w_benv(iter,:)     = w3(1,:);

    % 4) Ang & Farhang
    [~, ~, e4, w4] = ang_farhang(h, x, mu_0, rho, alpha, num_coef);
    error_ang(:,iter) = e4(:);
    w_ang(iter,:)     = w4(1,:);

    % 5) Matthews & Xie
    [~, ~, e5, w5] = matthews_xie(h, x, mu_0, rho, num_coef);
    error_mx(:,iter) = e5(:);
    w_mx(iter,:)     = w5(1,:);
end

% True coefficient
w_true = 0.9 * ones(1, N);

% Weight errors
werr_lms_001 = w_true - w_lms_001;
werr_lms_01  = w_true - w_lms_01;
werr_benv    = w_true - w_benv;
werr_ang     = w_true - w_ang;
werr_mx      = w_true - w_mx;

% Mean weight trajectories
mw_lms_001 = mean(w_lms_001, 1);
mw_lms_01  = mean(w_lms_01, 1);
mw_benv    = mean(w_benv, 1);
mw_ang     = mean(w_ang, 1);
mw_mx      = mean(w_mx, 1);

%% Plot: mean weight evolution
figure('Name','Part (a) - Weight Evolution','Color','w','Position',[80 80 1100 450]);
plot(mw_lms_001, 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(mw_lms_01,  'Color', c_orange, 'LineWidth', lw_main);
plot(mw_benv,    'Color', c_green,  'LineWidth', lw_main);
plot(mw_ang,     'Color', c_purple, 'LineWidth', lw_main);
plot(mw_mx,      'Color', c_red,    'LineWidth', lw_main);
yline(0.9, '--', 'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
xlabel('Sample','FontSize',11);
ylabel('Weight magnitude','FontSize',11);
legend('\mu = 0.01','\mu = 0.1','Benveniste','Ang & Farhang','Matthews & Xie','True value','Location','best');
title('Ensemble-averaged evolution of weight estimate $w_0$','Interpreter','latex','FontSize',12);

%% Plot: weight error curves
figure('Name','Part (a) - Weight Error Comparison','Color','w','Position',[60 60 1400 450]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile;
plot(mean(werr_lms_001,1), 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(mean(werr_lms_01,1),  'Color', c_orange, 'LineWidth', lw_main);
plot(mean(werr_benv,1),    'Color', c_green,  'LineWidth', lw_main);
plot(mean(werr_ang,1),     'Color', c_purple, 'LineWidth', lw_main);
plot(mean(werr_mx,1),      'Color', c_red,    'LineWidth', lw_main);
grid on; box on;
xlabel('Sample'); ylabel('Weight error');
title('Mean weight error');
legend('\mu = 0.01','\mu = 0.1','Benveniste','Ang & Farhang','Matthews & Xie','Location','best');
ylim([-0.1 0.9]);

nexttile;
plot(pow2db(mean(werr_lms_001.^2,1) + eps), 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(pow2db(mean(werr_lms_01.^2,1)  + eps), 'Color', c_orange, 'LineWidth', lw_main);
plot(pow2db(mean(werr_benv.^2,1)    + eps), 'Color', c_green,  'LineWidth', lw_main);
plot(pow2db(mean(werr_ang.^2,1)     + eps), 'Color', c_purple, 'LineWidth', lw_main);
plot(pow2db(mean(werr_mx.^2,1)      + eps), 'Color', c_red,    'LineWidth', lw_main);
grid on; box on;
xlabel('Sample'); ylabel('Squared weight error (dB)');
title('Mean squared weight error');
legend('\mu = 0.01','\mu = 0.1','Benveniste','Ang & Farhang','Matthews & Xie','Location','best');

nexttile;
load('method_1.mat'); % assumes method_1 is 3 x N
plot(method_1(1,:), 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(method_1(2,:), 'Color', c_orange, 'LineWidth', lw_main);
plot(method_1(3,:), 'Color', c_green,  'LineWidth', lw_main);
grid on; box on;
xlabel('Sample'); ylabel('Squared weight error (dB)');
title('Method 1 - idealised case');
legend('\mu = 0.01','\mu = 0.1','\mu = 1','Location','best');

%% Plot: squared prediction error
sqerr_lms_001 = pow2db(mean(error_lms_001.^2, 2) + eps);
sqerr_lms_01  = pow2db(mean(error_lms_01.^2,  2) + eps);
sqerr_benv    = pow2db(mean(error_benv.^2,    2) + eps);
sqerr_ang     = pow2db(mean(error_ang.^2,     2) + eps);
sqerr_mx      = pow2db(mean(error_mx.^2,      2) + eps);

figure('Name','Part (a) - Prediction Error','Color','w','Position',[100 100 1100 420]);
plot(sqerr_lms_001, 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(sqerr_lms_01,  'Color', c_orange, 'LineWidth', lw_main);
plot(sqerr_benv,    'Color', c_green,  'LineWidth', lw_main);
plot(sqerr_ang,     'Color', c_purple, 'LineWidth', lw_main);
plot(sqerr_mx,      'Color', c_red,    'LineWidth', lw_main);
grid on; box on;
xlabel('Sample','FontSize',11);
ylabel('Squared error (dB)','FontSize',11);
legend('\mu = 0.01','\mu = 0.1','Benveniste','Ang & Farhang','Matthews & Xie','Location','best');
title('Ensemble-averaged squared prediction error','FontSize',12);

%% ========================================================================
%% Part (c) - Benveniste vs GNGD
%% ========================================================================
Num_iter  = 100;
N         = 1000;
a         = 1;
b         = [1 0.9];
num_coef  = length(b);
noise_var = 0.5;

mu_0      = 0.1;           % Initial step size for Benveniste
mu_test   = [0.1 0.5];     % Initial step sizes for GNGD
rho       = 0.01;

figure('Name','Part (c) - Weight Evolution','Color','w','Position',[70 70 1200 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

figure('Name','Part (c) - Weight Error','Color','w','Position',[90 90 1200 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

for i = 1:length(mu_test)
    epsilon_0 = 1 / mu_test(i);

    error_benv = zeros(N, Num_iter);
    error_gngd = zeros(N, Num_iter);

    w_benv = zeros(Num_iter, N);
    w_gngd = zeros(Num_iter, N);

    for iter = 1:Num_iter
        h = sqrt(noise_var) * randn(N,1);
        x = filter(b, a, h);

        [~, ~, e_b, w_b] = benveniste(h, x, mu_0, rho, num_coef);
        [~, e_g, w_g]    = GNGD(h, x, mu_test(i), epsilon_0, rho, num_coef);

        error_benv(:,iter) = e_b(:);
        error_gngd(:,iter) = e_g(:);

        w_benv(iter,:) = w_b(1,:);
        w_gngd(iter,:) = w_g(1,:);
    end

    w0_benv = mean(w_benv, 1);
    w0_gngd = mean(w_gngd, 1);
    w_true  = 0.9 * ones(1, N);

    figure(findobj('Name','Part (c) - Weight Evolution'));
    nexttile;
    plot(w0_gngd, 'Color', c_blue,   'LineWidth', lw_main); hold on;
    plot(w0_benv, 'Color', c_orange, 'LineWidth', lw_main);
    yline(0.9, '--', 'Color', c_grey, 'LineWidth', lw_ref);
    grid on; box on;
    xlabel('Sample'); ylabel('Weight magnitude');
    title(['Weight evolution, \mu = ', num2str(mu_test(i))]);
    legend('GNGD','Benveniste','True value','Location','best');
    ylim([0 0.95]);

    figure(findobj('Name','Part (c) - Weight Error'));
    nexttile;
    plot(w_true - w0_benv, 'Color', c_orange, 'LineWidth', lw_main); hold on;
    plot(w_true - w0_gngd, 'Color', c_blue,   'LineWidth', lw_main);
    grid on; box on;
    xlabel('Sample'); ylabel('Weight error');
    title(['Weight error, \mu = ', num2str(mu_test(i))]);
    legend('Benveniste','GNGD','Location','best');
    ylim([0 0.9]);
end

%% ========================================================================
%% Parameter sweep 1: effect of rho
%% ========================================================================
N_iter      = 100;
rho_values  = 0.0001:0.0001:0.01;
mu_0        = 0.1;
mu_gngd     = 0.1;
epsilon_0   = 1 / mu_gngd;
threshold   = 0.001;

convergence_time_ben  = zeros(1, length(rho_values));
convergence_time_gngd = zeros(1, length(rho_values));

for r_idx = 1:length(rho_values)
    rho_now = rho_values(r_idx);

    w_benv = zeros(N_iter, N);
    w_gngd = zeros(N_iter, N);

    for iter = 1:N_iter
        h = sqrt(noise_var) * randn(N,1);
        x = filter(b, a, h);

        [~,~,~,wb] = benveniste_Method_1(h, x, mu_0, rho_now, num_coef);
        [~,~,wg]   = GNGD_Method_1(h, x, mu_gngd, epsilon_0, rho_now, num_coef);

        w_benv(iter,:) = wb(1,:);
        w_gngd(iter,:) = wg(1,:);
    end

    w_true = 0.9 * ones(1, N);

    werr_ben  = mean(abs(w_true - w_benv), 1);
    werr_gngd = mean(abs(w_true - w_gngd), 1);

    idx_ben = find(werr_ben < threshold, 1, 'first');
    idx_gngd = find(werr_gngd < threshold, 1, 'first');

    if isempty(idx_ben),  idx_ben = N;  end
    if isempty(idx_gngd), idx_gngd = N; end

    convergence_time_ben(r_idx)  = idx_ben;
    convergence_time_gngd(r_idx) = idx_gngd;
end

%% ========================================================================
%% Parameter sweep 2: effect of epsilon_0
%% ========================================================================
epsilon_values = 0.01:0.01:2;
rho            = 0.005;
mu_0           = 0.1;
mu_gngd        = 0.1;

conv_eps_gngd = zeros(1, length(epsilon_values));

for e_idx = 1:length(epsilon_values)
    eps_now = epsilon_values(e_idx);

    w_gngd = zeros(Num_iter, N);

    for iter = 1:Num_iter
        h = sqrt(noise_var) * randn(N,1);
        x = filter(b, a, h);

        [~,~,wg] = GNGD_Method_1(h, x, mu_gngd, eps_now, rho, num_coef);
        w_gngd(iter,:) = wg(1,:);
    end

    w_true = 0.9 * ones(1, N);
    w0_gngd = mean(w_gngd, 1);
    werr_gngd = abs(w0_gngd - w_true);

    idx_gngd = find(werr_gngd < threshold, 1, 'first');
    if isempty(idx_gngd), idx_gngd = N; end
    conv_eps_gngd(e_idx) = idx_gngd;
end

%% ========================================================================
%% Parameter sweep 3: effect of mu
%% ========================================================================
epsilon_0 = 0.01;
rho       = 0.005;
mu_values = 0.01:0.01:5;

conv_mu_ben  = zeros(1, length(mu_values));
conv_mu_gngd = zeros(1, length(mu_values));

for m_idx = 1:length(mu_values)
    mu_now = mu_values(m_idx);

    w_benv = zeros(Num_iter, N);
    w_gngd = zeros(Num_iter, N);

    for iter = 1:Num_iter
        h = sqrt(noise_var) * randn(N,1);
        x = filter(b, a, h);

        [~,~,~,wb] = benveniste_Method_1(h, x, mu_now, rho, num_coef);
        [~,~,wg]   = GNGD_Method_1(h, x, mu_now, epsilon_0, rho, num_coef);

        w_benv(iter,:) = wb(1,:);
        w_gngd(iter,:) = wg(1,:);
    end

    w_true = 0.9 * ones(1, N);
    w0_ben = mean(w_benv, 1);
    w0_gngd = mean(w_gngd, 1);

    werr_ben  = abs(w0_ben - w_true);
    werr_gngd = abs(w0_gngd - w_true);

    idx_ben = find(werr_ben < threshold, 1, 'first');
    idx_gngd = find(werr_gngd < threshold, 1, 'first');

    if isempty(idx_ben),  idx_ben = N;  end
    if isempty(idx_gngd), idx_gngd = N; end

    conv_mu_ben(m_idx)  = idx_ben;
    conv_mu_gngd(m_idx) = idx_gngd;
end

%% Plot parameter sweeps
figure('Name','Part (c) - Parameter Sensitivity','Color','w','Position',[60 60 1400 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile;
plot(rho_values, 1 ./ convergence_time_ben,  'Color', c_orange, 'LineWidth', lw_main); hold on;
plot(rho_values, 1 ./ convergence_time_gngd, 'Color', c_blue,   'LineWidth', lw_main);
grid on; box on;
xlabel('\rho value'); ylabel('Convergence speed');
title('Effect of \rho');
legend('Benveniste','GNGD','Location','best');

nexttile;
plot(epsilon_values, 1 ./ conv_eps_gngd, 'Color', c_blue, 'LineWidth', lw_main);
grid on; box on;
xlabel('\epsilon_0 value'); ylabel('Convergence speed');
title('Effect of \epsilon_0 on GNGD');
ylim([0 0.01]);

nexttile;
plot(mu_values, 1 ./ conv_mu_ben,  'Color', c_orange, 'LineWidth', lw_main); hold on;
plot(mu_values, 1 ./ conv_mu_gngd, 'Color', c_blue,   'LineWidth', lw_main);
grid on; box on;
xlabel('\mu value'); ylabel('Convergence speed');
title('Effect of \mu');
legend('Benveniste','GNGD','Location','best');
xlim([mu_values(1) mu_values(end)]);