%% Assignment 3.1(b) - CLMS and ACLMS for Wind Data
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

lw_main    = 1.6;
ms_scatter = 6;

rng('default');

%% ========================================================================
%% Load wind data
%% ========================================================================
high_wind   = load('high-wind.mat');
medium_wind = load('medium-wind.mat');
low_wind    = load('low-wind.mat');

%% ========================================================================
%% Form complex-valued wind signals: z(n) = v_east(n) + j v_north(n)
%% ========================================================================
z_high   = high_wind.v_east   + 1j * high_wind.v_north;
z_medium = medium_wind.v_east + 1j * medium_wind.v_north;
z_low    = low_wind.v_east    + 1j * low_wind.v_north;

%% ========================================================================
%% Circularity coefficients
%% ========================================================================
pseudo_high = mean(z_high.^2);
cov_high    = mean(abs(z_high).^2);
rho_high    = abs(pseudo_high) / cov_high;

pseudo_medium = mean(z_medium.^2);
cov_medium    = mean(abs(z_medium).^2);
rho_medium    = abs(pseudo_medium) / cov_medium;

pseudo_low = mean(z_low.^2);
cov_low    = mean(abs(z_low).^2);
rho_low    = abs(pseudo_low) / cov_low;

% Extra descriptive statistics
mean_high_real = mean(real(z_high));
mean_high_imag = mean(imag(z_high));
var_high_real  = var(real(z_high));
var_high_imag  = var(imag(z_high));
diff_var_high  = abs(var_high_real - var_high_imag);

mean_medium_real = mean(real(z_medium));
mean_medium_imag = mean(imag(z_medium));
var_medium_real  = var(real(z_medium));
var_medium_imag  = var(imag(z_medium));
diff_var_medium  = abs(var_medium_real - var_medium_imag);

mean_low_real = mean(real(z_low));
mean_low_imag = mean(imag(z_low));
var_low_real  = var(real(z_low));
var_low_imag  = var(imag(z_low));
diff_var_low  = abs(var_low_real - var_low_imag);

%% ========================================================================
%% Plot circularity plots for the three wind regimes
%% ========================================================================
figure('Name','Circularity of Wind Regimes', ...
       'Color','w','Position',[70 70 1350 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

% Reduce density slightly for plotting clarity
idx_high   = 1:2:length(z_high);
idx_medium = 1:2:length(z_medium);
idx_low    = 1:2:length(z_low);

nexttile;
scatter(real(z_high(idx_high)), imag(z_high(idx_high)), ms_scatter, c_blue, '.'); hold on;
scatter(mean_high_real, mean_high_imag, 120, c_green, 'filled');
grid on; box on; axis equal;
xlabel('Real part (East)');
ylabel('Imaginary part (North)');
title(['High-speed wind, |\rho| = ', num2str(rho_high,'%.4f')]);

nexttile;
scatter(real(z_medium(idx_medium)), imag(z_medium(idx_medium)), ms_scatter, c_orange, '.'); hold on;
scatter(mean_medium_real, mean_medium_imag, 120, c_green, 'filled');
grid on; box on; axis equal;
xlabel('Real part (East)');
ylabel('Imaginary part (North)');
title(['Medium-speed wind, |\rho| = ', num2str(rho_medium,'%.4f')]);

nexttile;
scatter(real(z_low(idx_low)), imag(z_low(idx_low)), ms_scatter, c_grey, '.'); hold on;
scatter(mean_low_real, mean_low_imag, 120, c_green, 'filled');
grid on; box on; axis equal;
xlabel('Real part (East)');
ylabel('Imaginary part (North)');
title(['Low-speed wind, |\rho| = ', num2str(rho_low,'%.4f')]);

%% ========================================================================
%% One-step prediction using CLMS and ACLMS
%% ========================================================================
mu = [0.001, 0.005, 0.1];   % [high, medium, low]
M_values = 1:30;            % Model orders to test

% Predictor inputs: x(n-1)
x_high   = [0; z_high(1:end-1)];
x_medium = [0; z_medium(1:end-1)];
x_low    = [0; z_low(1:end-1)];

% Preallocate squared error matrices: rows = model order, cols = time
err_CLMS_high   = zeros(length(M_values), length(z_high));
err_CLMS_medium = zeros(length(M_values), length(z_medium));
err_CLMS_low    = zeros(length(M_values), length(z_low));

err_ACLMS_high   = zeros(length(M_values), length(z_high));
err_ACLMS_medium = zeros(length(M_values), length(z_medium));
err_ACLMS_low    = zeros(length(M_values), length(z_low));

for m_idx = 1:length(M_values)
    M = M_values(m_idx);

    % CLMS
    [~, e_high_clms, ~]   = CLMS_3_1_b(x_high,   z_high,   mu(1), M);
    [~, e_medium_clms, ~] = CLMS_3_1_b(x_medium, z_medium, mu(2), M);
    [~, e_low_clms, ~]    = CLMS_3_1_b(x_low,    z_low,    mu(3), M);

    err_CLMS_high(m_idx,:)   = abs(e_high_clms).^2;
    err_CLMS_medium(m_idx,:) = abs(e_medium_clms).^2;
    err_CLMS_low(m_idx,:)    = abs(e_low_clms).^2;

    % ACLMS
    [~, e_high_aclms, ~, ~]   = ACLMS_3_1_b(x_high,   z_high,   mu(1), M);
    [~, e_medium_aclms, ~, ~] = ACLMS_3_1_b(x_medium, z_medium, mu(2), M);
    [~, e_low_aclms, ~, ~]    = ACLMS_3_1_b(x_low,    z_low,    mu(3), M);

    err_ACLMS_high(m_idx,:)   = abs(e_high_aclms).^2;
    err_ACLMS_medium(m_idx,:) = abs(e_medium_aclms).^2;
    err_ACLMS_low(m_idx,:)    = abs(e_low_aclms).^2;
end

%% ========================================================================
%% Compute MSPE curves versus model order
%% ========================================================================
mspe_CLMS_high   = pow2db(mean(err_CLMS_high,   2) + eps);
mspe_CLMS_medium = pow2db(mean(err_CLMS_medium, 2) + eps);
mspe_CLMS_low    = pow2db(mean(err_CLMS_low,    2) + eps);

mspe_ACLMS_high   = pow2db(mean(err_ACLMS_high,   2) + eps);
mspe_ACLMS_medium = pow2db(mean(err_ACLMS_medium, 2) + eps);
mspe_ACLMS_low    = pow2db(mean(err_ACLMS_low,    2) + eps);

%% ========================================================================
%% Plot MSPE versus model order
%% ========================================================================
figure('Name','MSPE vs Model Order for Wind Data', ...
       'Color','w','Position',[70 70 1350 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile;
plot(M_values, mspe_CLMS_high,   'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(M_values, mspe_ACLMS_high,  'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Model order M');
ylabel('MSPE (dB)');
title(['High-speed wind, \mu = ', num2str(mu(1))]);
legend('CLMS','ACLMS','Location','best');

nexttile;
plot(M_values, mspe_CLMS_medium,  'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(M_values, mspe_ACLMS_medium, 'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Model order M');
ylabel('MSPE (dB)');
title(['Medium-speed wind, \mu = ', num2str(mu(2))]);
legend('CLMS','ACLMS','Location','best');

nexttile;
plot(M_values, mspe_CLMS_low,   'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(M_values, mspe_ACLMS_low,  'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Model order M');
ylabel('MSPE (dB)');
title(['Low-speed wind, \mu = ', num2str(mu(3))]);
legend('CLMS','ACLMS','Location','best');

%% ========================================================================
%% Minimum MSPE and percentage improvement of ACLMS over CLMS
%% ========================================================================
[min_CLMS_high,   idx_CLMS_high]   = min(mspe_CLMS_high);
[min_ACLMS_high,  idx_ACLMS_high]  = min(mspe_ACLMS_high);
perc_diff_high = abs((min_CLMS_high - min_ACLMS_high) / min_CLMS_high) * 100;

[min_CLMS_medium,  idx_CLMS_medium]  = min(mspe_CLMS_medium);
[min_ACLMS_medium, idx_ACLMS_medium] = min(mspe_ACLMS_medium);
perc_diff_medium = abs((min_CLMS_medium - min_ACLMS_medium) / min_CLMS_medium) * 100;

[min_CLMS_low,   idx_CLMS_low]   = min(mspe_CLMS_low);
[min_ACLMS_low,  idx_ACLMS_low]  = min(mspe_ACLMS_low);
perc_diff_low = abs((min_CLMS_low - min_ACLMS_low) / min_CLMS_low) * 100;

%% ========================================================================
%% Choose representative model orders for visual comparison
%% ========================================================================
M_high   = 4;
M_medium = 5;
M_low    = 6;

% CLMS estimates
[zhat_CLMS_high,   ~, ~] = CLMS_3_1_b(x_high,   z_high,   mu(1), M_high);
[zhat_CLMS_medium, ~, ~] = CLMS_3_1_b(x_medium, z_medium, mu(2), M_medium);
[zhat_CLMS_low,    ~, ~] = CLMS_3_1_b(x_low,    z_low,    mu(3), M_low);

% ACLMS estimates
[zhat_ACLMS_high,   ~, ~, ~] = ACLMS_3_1_b(x_high,   z_high,   mu(1), M_high);
[zhat_ACLMS_medium, ~, ~, ~] = ACLMS_3_1_b(x_medium, z_medium, mu(2), M_medium);
[zhat_ACLMS_low,    ~, ~, ~] = ACLMS_3_1_b(x_low,    z_low,    mu(3), M_low);

%% ========================================================================
%% Plot scatter comparison for chosen model orders
%% ========================================================================
figure('Name','Wind Data: Original vs CLMS/ACLMS Estimates', ...
       'Color','w','Position',[60 60 1200 900]);
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

% Reduce density for scatter plotting
idx_high_scatter   = 1:2:length(z_high);
idx_medium_scatter = 1:2:length(z_medium);
idx_low_scatter    = 1:2:length(z_low);

% High wind
nexttile;
scatter(real(z_high(idx_high_scatter)), imag(z_high(idx_high_scatter)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(zhat_CLMS_high(idx_high_scatter)), imag(zhat_CLMS_high(idx_high_scatter)), ...
        ms_scatter, c_blue, '.');
grid on; box on; axis equal;
ylabel('Imaginary part (North)');
title(['High-speed wind, CLMS, M = ', num2str(M_high)]);
legend('Original','CLMS estimate','Location','best');

nexttile;
scatter(real(z_high(idx_high_scatter)), imag(z_high(idx_high_scatter)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(zhat_ACLMS_high(idx_high_scatter)), imag(zhat_ACLMS_high(idx_high_scatter)), ...
        ms_scatter, c_orange, '.');
grid on; box on; axis equal;
ylabel('Imaginary part (North)');
title(['High-speed wind, ACLMS, M = ', num2str(M_high)]);
legend('Original','ACLMS estimate','Location','best');

% Medium wind
nexttile;
scatter(real(z_medium(idx_medium_scatter)), imag(z_medium(idx_medium_scatter)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(zhat_CLMS_medium(idx_medium_scatter)), imag(zhat_CLMS_medium(idx_medium_scatter)), ...
        ms_scatter, c_blue, '.');
grid on; box on; axis equal;
ylabel('Imaginary part (North)');
title(['Medium-speed wind, CLMS, M = ', num2str(M_medium)]);
legend('Original','CLMS estimate','Location','best');

nexttile;
scatter(real(z_medium(idx_medium_scatter)), imag(z_medium(idx_medium_scatter)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(zhat_ACLMS_medium(idx_medium_scatter)), imag(zhat_ACLMS_medium(idx_medium_scatter)), ...
        ms_scatter, c_orange, '.');
grid on; box on; axis equal;
ylabel('Imaginary part (North)');
title(['Medium-speed wind, ACLMS, M = ', num2str(M_medium)]);
legend('Original','ACLMS estimate','Location','best');

% Low wind
nexttile;
scatter(real(z_low(idx_low_scatter)), imag(z_low(idx_low_scatter)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(zhat_CLMS_low(idx_low_scatter)), imag(zhat_CLMS_low(idx_low_scatter)), ...
        ms_scatter, c_blue, '.');
grid on; box on; axis equal;
xlabel('Real part (East)');
ylabel('Imaginary part (North)');
title(['Low-speed wind, CLMS, M = ', num2str(M_low)]);
legend('Original','CLMS estimate','Location','best');

nexttile;
scatter(real(z_low(idx_low_scatter)), imag(z_low(idx_low_scatter)), ...
        ms_scatter, c_grey, '.'); hold on;
scatter(real(zhat_ACLMS_low(idx_low_scatter)), imag(zhat_ACLMS_low(idx_low_scatter)), ...
        ms_scatter, c_orange, '.');
grid on; box on; axis equal;
xlabel('Real part (East)');
ylabel('Imaginary part (North)');
title(['Low-speed wind, ACLMS, M = ', num2str(M_low)]);
legend('Original','ACLMS estimate','Location','best');

%% ========================================================================
%% Print summary to command window
%% ========================================================================
fprintf('\n================ Assignment 3.1(b) Summary ================\n');

fprintf('\nCircularity coefficients:\n');
fprintf('High-speed wind   : %.6f\n', rho_high);
fprintf('Medium-speed wind : %.6f\n', rho_medium);
fprintf('Low-speed wind    : %.6f\n', rho_low);

fprintf('\nMinimum MSPE values (dB):\n');
fprintf('High-speed wind   | CLMS: %.4f at M=%d | ACLMS: %.4f at M=%d | improvement: %.2f%%\n', ...
    min_CLMS_high, idx_CLMS_high, min_ACLMS_high, idx_ACLMS_high, perc_diff_high);
fprintf('Medium-speed wind | CLMS: %.4f at M=%d | ACLMS: %.4f at M=%d | improvement: %.2f%%\n', ...
    min_CLMS_medium, idx_CLMS_medium, min_ACLMS_medium, idx_ACLMS_medium, perc_diff_medium);
fprintf('Low-speed wind    | CLMS: %.4f at M=%d | ACLMS: %.4f at M=%d | improvement: %.2f%%\n', ...
    min_CLMS_low, idx_CLMS_low, min_ACLMS_low, idx_ACLMS_low, perc_diff_low);

fprintf('\nVariance difference between real and imaginary parts:\n');
fprintf('High-speed wind   : %.6f\n', diff_var_high);
fprintf('Medium-speed wind : %.6f\n', diff_var_medium);
fprintf('Low-speed wind    : %.6f\n', diff_var_low);