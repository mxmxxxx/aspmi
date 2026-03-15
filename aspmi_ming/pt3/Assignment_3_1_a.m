%% Assignment 3.1(a) - Circularity and CLMS vs ACLMS
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

lw_main = 1.6;
ms_scatter = 8;

rng('default');

%% ========================================================================
%% Part 1 - Circularity of WGN and WLMA(1)
%% ========================================================================
N = 100000;   % Signal length for circularity illustration

% Generate circular complex white Gaussian noise
x = randn(1,N) + 1j*randn(1,N);

% Generate WLMA(1) process:
% y(n) = x(n) + b1*x(n-1) + b2*conj(x(n-1))
b = [1, (1.5 + 1j), (2.5 - 0.5j)];
y = zeros(1,N);
y(1) = x(1);

for n = 1:N-1
    y(n+1) = b(1)*x(n+1) + b(2)*x(n) + b(3)*conj(x(n));
end

% Circularity coefficient for WGN
pseudoCov_x = mean(x.^2);
cov_x       = mean(abs(x).^2);
rho_x       = abs(pseudoCov_x) / cov_x;

% Circularity coefficient for WLMA(1)
pseudoCov_y = mean(y.^2);
cov_y       = mean(abs(y).^2);
rho_y       = abs(pseudoCov_y) / cov_y;

% Plot circularity scatter
figure('Name','Circularity of WGN and WLMA(1)', ...
       'Color','w','Position',[100 100 1200 480]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% To avoid overplotting, show only a subset
idx_plot = 1:20:N;

nexttile;
scatter(real(x(idx_plot)), imag(x(idx_plot)), ms_scatter, c_blue, '.');
grid on; box on; axis equal;
xlabel('Real part');
ylabel('Imaginary part');
title(['Circular WGN, |\rho| = ', num2str(rho_x,'%.4f')]);

nexttile;
scatter(real(y(idx_plot)), imag(y(idx_plot)), ms_scatter, c_red, '.');
grid on; box on; axis equal;
xlabel('Real part');
ylabel('Imaginary part');
title(['WLMA(1), |\rho| = ', num2str(rho_y,'%.4f')]);

%% ========================================================================
%% Part 2 - CLMS vs ACLMS over multiple realisations
%% ========================================================================
N_iter   = 100;     % Number of Monte Carlo trials
N        = 1000;    % Signal length
mu       = 0.03;    % Learning rate
num_coef = length(b) - 1;

% Preallocate error matrices
error_CLMS  = zeros(N_iter, N);
error_ACLMS = zeros(N_iter, N);

% Save one representative realisation for scatter plots
signal_demo      = [];
signal_est_CLMS  = [];
signal_est_ACLMS = [];

for iter = 1:N_iter
    % Circular complex white Gaussian noise
    x = randn(1,N) + 1j*randn(1,N);

    % WLMA(1) process
    y = zeros(1,N);
    y(1) = x(1);
    for n = 1:N-1
        y(n+1) = b(1)*x(n+1) + b(2)*x(n) + b(3)*conj(x(n));
    end

    % CLMS
    [yhat_CLMS, err_CLMS, ~] = CLMS_MA(x, y, mu, num_coef);
    error_CLMS(iter,:) = err_CLMS;

    % ACLMS
    [yhat_ACLMS, err_ACLMS, ~, ~] = ACLMS_MA(x, y, mu, num_coef);
    error_ACLMS(iter,:) = err_ACLMS;

    % Save one realisation for visual comparison
    if iter == 1
        signal_demo      = y;
        signal_est_CLMS  = yhat_CLMS;
        signal_est_ACLMS = yhat_ACLMS;
    end
end

% Ensemble-averaged learning curves
lc_CLMS  = pow2db(mean(abs(error_CLMS).^2, 1) + eps);
lc_ACLMS = pow2db(mean(abs(error_ACLMS).^2, 1) + eps);

%% ========================================================================
%% Plot learning curves and one-realisation constellation comparison
%% ========================================================================
figure('Name','CLMS vs ACLMS on WLMA(1)', ...
       'Color','w','Position',[80 80 1350 430]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

% Learning curves
nexttile;
plot(lc_CLMS,  'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(lc_ACLMS, 'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Sample index');
ylabel('Squared error (dB)');
title('Ensemble-averaged learning curves');
legend('CLMS','ACLMS','Location','best');

% CLMS scatter comparison
nexttile;
idx_demo = 1:2:N; % reduce density a bit
scatter(real(signal_demo(idx_demo)),     imag(signal_demo(idx_demo)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(signal_est_CLMS(idx_demo)), imag(signal_est_CLMS(idx_demo)), ...
        ms_scatter, c_blue, '.');
grid on; box on; axis equal;
xlabel('Real part');
ylabel('Imaginary part');
title('One realisation: CLMS');
legend('Original process','CLMS estimate','Location','best');

% ACLMS scatter comparison
nexttile;
scatter(real(signal_demo(idx_demo)),      imag(signal_demo(idx_demo)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(signal_est_ACLMS(idx_demo)), imag(signal_est_ACLMS(idx_demo)), ...
        ms_scatter, c_orange, '.');
grid on; box on; axis equal;
xlabel('Real part');
ylabel('Imaginary part');
title('One realisation: ACLMS');
legend('Original process','ACLMS estimate','Location','best');

%% ========================================================================
%% Print summary to command window
%% ========================================================================
fprintf('\n================ Assignment 3.1(a) Summary ================\n');
fprintf('Circularity coefficient of WGN     : %.6f\n', rho_x);
fprintf('Circularity coefficient of WLMA(1) : %.6f\n', rho_y);
fprintf('Final mean squared error (CLMS)    : %.6f dB\n', lc_CLMS(end));
fprintf('Final mean squared error (ACLMS)   : %.6f dB\n', lc_ACLMS(end));