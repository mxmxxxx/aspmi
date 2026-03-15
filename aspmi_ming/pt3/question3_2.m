%% Assignment 3.2 - Adaptive AR Model Based Time-Frequency Estimation
close all; clear; clc;

%% ========================================================================
%% Global plotting style
%% ========================================================================
c_blue   = [0.00 0.45 0.74];
c_orange = [0.85 0.33 0.10];
c_purple = [0.49 0.18 0.56];
c_red    = [0.80 0.20 0.20];
c_grey   = [0.35 0.35 0.35];

lw_main = 1.6;

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

rng('default');

%% ========================================================================
%% Part (a) - Generate FM signal
%% ========================================================================
N  = 1500;     % Number of samples
fs = 2000;     % Sampling frequency
sigma2_eta = 0.05;

n = 1:N;

% Circular complex white noise
eta = sqrt(sigma2_eta)*randn(1,N) + 1j*sqrt(sigma2_eta)*randn(1,N);

% Instantaneous frequency f(n)
f_n = [100*ones(1,500), ...
       100 + ((501:1000)-500)/2, ...
       100 + (((1001:1500)-1000)/25).^2];

% Phase obtained by integration
phi = cumtrapz(f_n);

% FM signal
y = exp(1j*(2*pi/fs)*phi) + eta;

%% ========================================================================
%% Plot frequency and phase
%% ========================================================================
figure('Name','FM Signal Definition','Color','w','Position',[100 100 1200 420]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(n, f_n, 'Color', c_red, 'LineWidth', lw_main);
grid on; box on;
xlabel('Time index $n$');
ylabel('Frequency (Hz)');
title('Instantaneous frequency $f(n)$');

nexttile;
plot(n, wrapTo2Pi(phi), 'Color', c_blue, 'LineWidth', lw_main);
grid on; box on;
xlabel('Time index $n$');
ylabel('Phase (rad)');
title('Phase $\phi(n)$');
xlim([1 200]);

%% ========================================================================
%% AR modelling of the full FM signal
%% ========================================================================
orders = [1, 5, 10];
colors = {c_blue, c_orange, c_purple};

figure('Name','AR Power Spectra of Full FM Signal', ...
       'Color','w','Position',[80 80 1300 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

for k = 1:length(orders)
    p = orders(k);

    % Yule-Walker AR model
    a = aryule(y, p);

    % Frequency response and PSD
    [h, w] = freqz(1, a, N, fs);
    psd = 10*log10(abs(h).^2 + eps);

    nexttile;
    plot(w, psd, 'Color', colors{k}, 'LineWidth', lw_main);
    grid on; box on;
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    title(['AR(', num2str(p), ')']);
end

%% ========================================================================
%% Piecewise AR(1) on 3 equal segments
%% ========================================================================
figure('Name','Piecewise AR(1) Modelling', ...
       'Color','w','Position',[100 100 1300 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

seg_len = 500;

for seg = 1:3
    idx = (seg_len*(seg-1)+1):(seg_len*seg);
    y_seg = y(idx);

    a_seg = aryule(y_seg, 1);
    [h_seg, w_seg] = freqz(1, a_seg, seg_len, fs);
    psd_seg = 10*log10(abs(h_seg).^2 + eps);

    nexttile;
    plot(w_seg, psd_seg, 'Color', c_blue, 'LineWidth', lw_main);
    grid on; box on;
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    title(['AR(1), segment ', num2str(seg)]);
end

%% ========================================================================
%% Windowed AR(1) attempt for local estimation
%% ========================================================================
window_len = 3;
fs_window = 10000;   % kept from your original code for local freq grid
allFreqEst = [];

for i = 1:floor(N/window_len)
    for j = 1:window_len
        try
            idx_start = window_len*(i-1) + j;
            idx_end   = window_len*i + (j-1);

            a_local = aryule(y(idx_start:idx_end), 1);
            [h_local, w_local] = freqz(1, a_local, floor(N/window_len), fs_window);
            psd_local = 10*log10(abs(h_local).^2 + eps);

            [~, maxIdx] = max(psd_local);
            allFreqEst = [allFreqEst, w_local(maxIdx)];
        catch
        end
    end
end

figure('Name','Windowed AR(1) Frequency Estimate', ...
       'Color','w','Position',[100 100 1000 420]);
plot(allFreqEst, 'Color', c_blue, 'LineWidth', 1.2); hold on;
plot(f_n, '--', 'Color', c_grey, 'LineWidth', lw_main);
grid on; box on;
xlabel('Time index $n$');
ylabel('Frequency (Hz)');
title('Windowed AR(1) estimate of instantaneous frequency');
legend('Windowed AR(1) estimate','True frequency','Location','best');

%% ========================================================================
%% Part (b) - CLMS-based time-frequency estimation
%% ========================================================================
% Regenerate signal
N  = 1500;
fs = 2000;
n = 1:N;

eta = sqrt(sigma2_eta)*randn(1,N) + 1j*sqrt(sigma2_eta)*randn(1,N);

f_n = [100*ones(1,500), ...
       100 + ((501:1000)-500)/2, ...
       100 + (((1001:1500)-1000)/25).^2];

phi = cumtrapz(f_n);
y = exp(1j*(2*pi/fs)*phi) + eta;

% One-step predictor input
x = [0, y(1:end-1)];

mu_values = [0.005, 0.05, 0.1, 0.5];
L = 1024;  % Number of frequency bins

figure('Name','CLMS Time-Frequency Estimation', ...
       'Color','w','Position',[50 50 1300 900]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

for k = 1:length(mu_values)
    mu = mu_values(k);

    % CLMS estimate of AR(1) coefficient
    [a_hat, e] = clms_b(x, y, mu, 1); %#ok<NASGU>

    H = zeros(L, N);

    for t_idx = 1:N
        [h_tf, w_tf] = freqz(1, [1; -conj(a_hat(t_idx))], L);
        H(:, t_idx) = abs(h_tf).^2;
    end

    % Robust clipping of large outliers for plotting
    H_clip = H;
    clip_level = 50 * median(H(:));
    H_clip(H_clip > clip_level) = clip_level;

    nexttile;
    surf(1:N, (w_tf*fs)/(2*pi), H_clip, 'LineStyle', 'none');
    view(2);
    axis tight;
    colormap turbo;
    colorbar('TickLabelInterpreter','latex');
    xlabel('Time index $n$');
    ylabel('Frequency (Hz)');
    title(['CLMS time-frequency estimate, $\mu = ', num2str(mu), '$']);
end