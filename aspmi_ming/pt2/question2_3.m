%% Assignment 2.3 - Adaptive Noise Cancellation / ALE
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

lw_main = 1.5;
lw_ref  = 1.2;

rng('default');

%% ========================================================================
%% Signal and noise model
%% ========================================================================
N = 10000;                    % Number of samples
n = (0:N-1)';                % Sample index
omega_0 = 0.01*pi;           % Sinusoid frequency
x = sin(omega_0*n);          % Clean sinusoidal signal

a = 1;                       % AR denominator for colouring filter
b = [1 0 0.5];               % eta(n)=v(n)+0.5v(n-2)

n_ss = 500:N;                % Steady-state region for MSPE

%% ========================================================================
%% Part (a) - Verify minimum delay
%% ========================================================================
mu = 0.01;
M  = 5;
N_iter = 100;
delay_values = 1:4;

MSPE_delay_mean = zeros(length(delay_values),1);

figure('Name','Part (a) - Minimum Delay Verification', ...
       'Color','w','Position',[80 80 1400 380]);
tiledlayout(1,length(delay_values),'Padding','compact','TileSpacing','compact');

for d_idx = 1:length(delay_values)
    Delta = delay_values(d_idx);

    MSPE_trials = zeros(N_iter,1);
    xhat_trials = zeros(N_iter,N);
    s_demo = [];

    for iter = 1:N_iter
        v = randn(N,1);
        eta = filter(b,a,v);
        s = x + eta;

        [x_hat,~,~] = ALE_lms(s, mu, M, Delta);

        if iter == 1
            s_demo = s;
        end

        xhat_trials(iter,:) = x_hat(:).';
        MSPE_trials(iter) = mean((x(n_ss)-x_hat(n_ss)).^2);
    end

    MSPE_delay_mean(d_idx) = mean(MSPE_trials);
    xhat_mean = mean(xhat_trials,1);

    nexttile;
    plot(n, s_demo, 'Color', c_blue,   'LineWidth', 0.8); hold on;
    plot(n, xhat_mean, 'Color', c_red, 'LineWidth', lw_main);
    plot(n, x, 'Color', c_grey, 'LineWidth', lw_ref);
    grid on; box on;
    ylim([-4 4]);
    xlabel('Sample index');
    ylabel('Amplitude');
    title(['\Delta = ', num2str(Delta), ...
           ', MSPE = ', num2str(MSPE_delay_mean(d_idx), '%.4f')]);
    legend('Noisy signal','ALE estimate','Clean signal','Location','best');
end

fprintf('\n================ Part (a): Minimum delay check ================\n');
for d_idx = 1:length(delay_values)
    fprintf('Delta = %d, mean steady-state MSPE = %.6f\n', ...
        delay_values(d_idx), MSPE_delay_mean(d_idx));
end

%% ========================================================================
%% Part (b1) - Effect of delay on MSPE
%% ========================================================================
mu = 0.01;
M_values = [5 10 15 20];
delay_values = 1:25;
N_iter = 100;

MSPE_vs_delay = zeros(length(M_values), length(delay_values));
xhat_plot = zeros(2,N);  % For Delta = 3 and 25 when M = 5

for m_idx = 1:length(M_values)
    M = M_values(m_idx);

    for d_idx = 1:length(delay_values)
        Delta = delay_values(d_idx);
        MSPE_trials = zeros(N_iter,1);
        xhat_trials = zeros(N_iter,N);

        for iter = 1:N_iter
            v = randn(N,1);
            eta = filter(b,a,v);
            s = x + eta;

            [x_hat,~,~] = ALE_lms(s, mu, M, Delta);

            xhat_trials(iter,:) = x_hat(:).';
            MSPE_trials(iter) = mean((x(n_ss)-x_hat(n_ss)).^2);
        end

        MSPE_vs_delay(m_idx, d_idx) = mean(MSPE_trials);

        if M == 5 && Delta == 3
            xhat_plot(1,:) = mean(xhat_trials,1);
        elseif M == 5 && Delta == 25
            xhat_plot(2,:) = mean(xhat_trials,1);
        end
    end
end

figure('Name','Part (b1) - Effect of Delay', ...
       'Color','w','Position',[80 80 1400 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile;
plot(delay_values, MSPE_vs_delay(1,:), 'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(delay_values, MSPE_vs_delay(2,:), 'Color', c_orange, 'LineWidth', lw_main);
plot(delay_values, MSPE_vs_delay(3,:), 'Color', c_green,  'LineWidth', lw_main);
plot(delay_values, MSPE_vs_delay(4,:), 'Color', c_purple, 'LineWidth', lw_main);
grid on; box on;
xlabel('Delay \Delta');
ylabel('MSPE');
title('Effect of delay on MSPE');
legend('M = 5','M = 10','M = 15','M = 20','Location','best');

nexttile;
plot(n, xhat_plot(1,:), 'Color', c_red,  'LineWidth', lw_main); hold on;
plot(n, x,             'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
xlabel('Sample index');
ylabel('Amplitude');
title('\Delta = 3, M = 5');
legend('\hat{x}(n)','x(n)','Location','best');

nexttile;
plot(n, xhat_plot(2,:), 'Color', c_red,  'LineWidth', lw_main); hold on;
plot(n, x,             'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
xlabel('Sample index');
ylabel('Amplitude');
title('\Delta = 25, M = 5');
legend('\hat{x}(n)','x(n)','Location','best');

%% ========================================================================
%% Part (b2) - Effect of filter order on MSPE
%% ========================================================================
mu = 0.001;
M_sweep = 1:20;
Delta = 3;
N_iter = 100;

MSPE_vs_M = zeros(length(M_sweep),1);

for m_idx = 1:length(M_sweep)
    M = M_sweep(m_idx);
    MSPE_trials = zeros(N_iter,1);

    for iter = 1:N_iter
        v = randn(N,1);
        eta = filter(b,a,v);
        s = x + eta;

        [x_hat,~,~] = ALE_lms(s, mu, M, Delta);
        MSPE_trials(iter) = mean((x(n_ss)-x_hat(n_ss)).^2);
    end

    MSPE_vs_M(m_idx) = mean(MSPE_trials);
end

[MSPE_min, idx_opt] = min(MSPE_vs_M);
M_opt = M_sweep(idx_opt);

figure('Name','Part (b2) - Effect of Filter Order', ...
       'Color','w','Position',[120 120 850 420]);
plot(M_sweep, MSPE_vs_M, 'Color', c_blue, 'LineWidth', lw_main); hold on;
stem(M_opt, MSPE_min, 'Color', c_red, 'LineWidth', 1.2, 'MarkerFaceColor', c_red);
grid on; box on;
xlabel('Filter order M');
ylabel('MSPE');
title('Effect of filter order on MSPE');
legend('MSPE','Minimum','Location','best');

fprintf('\n================ Part (b): Filter order sweep ================\n');
fprintf('Optimal M in sweep 1:20 is M = %d with MSPE = %.6f\n', M_opt, MSPE_min);

%% ========================================================================
%% Part (c) - Compare ALE and ANC
%% ========================================================================
mu = 0.01;
M = 5;
Delta = 3;
N_iter = 100;

xhat_ALE_trials = zeros(N_iter,N);
xhat_ANC_trials = zeros(N_iter,N);
MSPE_ALE_trials = zeros(N_iter,1);
MSPE_ANC_trials = zeros(N_iter,1);

s_demo = [];

for iter = 1:N_iter
    v = randn(N,1);
    eta = filter(b,a,v);
    s = x + eta;

    % Secondary reference correlated with the coloured noise
    u = 0.7*eta + 0.1;

    if iter == 1
        s_demo = s;
    end

    % ALE
    [xhat_ALE,~,~] = ALE_lms(s, mu, M, Delta);
    xhat_ALE_trials(iter,:) = xhat_ALE(:).';
    MSPE_ALE_trials(iter) = mean((x(n_ss)-xhat_ALE(n_ss)).^2);

    % ANC
    [~, xhat_ANC, ~] = ANC_lms(u, s, mu, M);
    xhat_ANC_trials(iter,:) = xhat_ANC(:).';
    MSPE_ANC_trials(iter) = mean((x(n_ss)-xhat_ANC(n_ss)).^2);
end

xhat_ALE_mean = mean(xhat_ALE_trials,1);
xhat_ANC_mean = mean(xhat_ANC_trials,1);

MSPE_ALE = mean(MSPE_ALE_trials);
MSPE_ANC = mean(MSPE_ANC_trials);

figure('Name','Part (c) - ALE vs ANC', ...
       'Color','w','Position',[80 80 1300 700]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(n, s_demo,       'Color', c_blue, 'LineWidth', 0.8); hold on;
plot(n, xhat_ALE_mean,'Color', c_red,  'LineWidth', lw_main);
plot(n, x,           'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
ylim([-4 4]);
xlabel('Sample index');
ylabel('Amplitude');
title(['ALE, MSPE = ', num2str(MSPE_ALE,'%.4f'), ...
       ', M = ', num2str(M), ', \Delta = ', num2str(Delta)]);
legend('Noisy signal','ALE estimate','Clean signal','Location','best');

nexttile;
plot(n, s_demo,       'Color', c_blue, 'LineWidth', 0.8); hold on;
plot(n, xhat_ANC_mean,'Color', c_red,  'LineWidth', lw_main);
plot(n, x,           'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
ylim([-4 4]);
xlabel('Sample index');
ylabel('Amplitude');
title(['ANC, MSPE = ', num2str(MSPE_ANC,'%.4f'), ...
       ', M = ', num2str(M)]);
legend('Noisy signal','ANC estimate','Clean signal','Location','best');

nexttile;
plot(n, xhat_ALE_mean, 'Color', c_red,  'LineWidth', lw_main); hold on;
plot(n, x,            'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
xlabel('Sample index');
ylabel('Amplitude');
title('Ensemble-averaged ALE estimate');
legend('ALE estimate','Clean signal','Location','best');

nexttile;
plot(n, xhat_ANC_mean, 'Color', c_red,  'LineWidth', lw_main); hold on;
plot(n, x,            'Color', c_grey, 'LineWidth', lw_ref);
grid on; box on;
xlabel('Sample index');
ylabel('Amplitude');
title('Ensemble-averaged ANC estimate');
legend('ANC estimate','Clean signal','Location','best');

fprintf('\n================ Part (c): ALE vs ANC ================\n');
fprintf('ALE mean steady-state MSPE = %.6f\n', MSPE_ALE);
fprintf('ANC mean steady-state MSPE = %.6f\n', MSPE_ANC);