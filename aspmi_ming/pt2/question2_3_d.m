%% Assignment 2.3(d) - EEG Denoising Using ANC
close all; clear; clc;

%% ========================================================================
%% Global plotting style
%% ========================================================================
c_blue   = [0.00 0.45 0.74];
c_orange = [0.85 0.33 0.10];
c_red    = [0.80 0.20 0.20];
c_grey   = [0.35 0.35 0.35];

lw_main = 1.5;
lw_ref  = 1.2;

rng('default');

%% ========================================================================
%% Load EEG data
%% ========================================================================
load('EEG_Data_Assignment2.mat');

% Choose channel: Cz or POz
data = Cz;   % change to POz if required

dt = 1/fs;                         % Sampling interval
N = length(data);                  % Number of samples
t = (0:N-1)' * dt;                 % Time axis in seconds

%% ========================================================================
%% Construct synthetic 50 Hz reference input
%% ========================================================================
f0 = 50;                           % Power-line interference frequency (Hz)
noise_var = 0.005;                 % Reference noise variance
ref_noise = sqrt(noise_var) * randn(N,1);
ref_signal = sin(2*pi*f0*t) + ref_noise;

%% ========================================================================
%% Spectrogram parameters
%% ========================================================================
L = 5 * fs;                        % Window length: 5 seconds
overlap_ratio = 0.5;               % 50% overlap
nOverlap = round(overlap_ratio * L);
nfft = 3 * L;

%% ========================================================================
%% Plot spectrogram of original EEG
%% ========================================================================
figure('Name','Original EEG Spectrogram', ...
       'Color','w','Position',[100 100 900 500]);
spectrogram(data, hanning(L), nOverlap, nfft, fs, 'yaxis');
ylim([0 60]);
title('Spectrogram of noise-corrupted EEG signal');
colormap turbo;

%% ========================================================================
%% Parameter sweep: effect of M and mu
%% ========================================================================
mu_values = [0.001, 0.01, 0.1];
M_values  = [2, 15, 30];

figure('Name','ANC Spectrograms for Different M and \mu', ...
       'Color','w','Position',[60 60 1300 900]);
tiledlayout(length(M_values), length(mu_values), ...
            'Padding','compact', 'TileSpacing','compact');

for m_idx = 1:length(M_values)
    M = M_values(m_idx);

    for mu_idx = 1:length(mu_values)
        mu = mu_values(mu_idx);

        % ANC
        [~, x_hat, ~] = ANC_lms(ref_signal, data, mu, M);

        nexttile;
        spectrogram(x_hat, hanning(L), nOverlap, nfft, fs, 'yaxis');
        ylim([0 60]);
        title(['M = ', num2str(M), ', \mu = ', num2str(mu)]);
        colormap turbo;
    end
end

%% ========================================================================
%% Choose optimal parameters
%% ========================================================================
mu_opt = 0.001;
M_opt  = 15;

[~, x_hat_opt, ~] = ANC_lms(ref_signal, data, mu_opt, M_opt);

%% ========================================================================
%% Compare original and denoised spectrograms
%% ========================================================================
figure('Name','Original vs Denoised EEG Spectrogram', ...
       'Color','w','Position',[100 100 1200 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
spectrogram(data, hanning(L), nOverlap, nfft, fs, 'yaxis');
ylim([0 60]);
title('Noise-corrupted EEG signal');
colormap turbo;

nexttile;
spectrogram(x_hat_opt, hanning(L), nOverlap, nfft, fs, 'yaxis');
ylim([0 60]);
title(['De-noised EEG signal (M = ', num2str(M_opt), ...
       ', \mu = ', num2str(mu_opt), ')']);
colormap turbo;

%% ========================================================================
%% PSD / Periodogram comparison
%% ========================================================================
% Reduce frequency resolution for clearer PSD comparison
N_psd = floor(N / 16);

% 10-second Welch window
win_len = round(10 / dt);

[psd_orig, f_orig] = pwelch(data, rectwin(win_len), 0, N_psd, fs, 'onesided');
[psd_denoised, f_denoised] = pwelch(x_hat_opt(500:end), rectwin(win_len), 0, N_psd, fs, 'onesided');

psd_orig_dB = pow2db(psd_orig + eps);
psd_denoised_dB = pow2db(psd_denoised + eps);

%% ========================================================================
%% Plot PSD comparison and PSD difference
%% ========================================================================
figure('Name','PSD Comparison Before and After ANC', ...
       'Color','w','Position',[100 100 1200 420]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(f_orig, psd_orig_dB,     'Color', c_blue,   'LineWidth', lw_main); hold on;
plot(f_denoised, psd_denoised_dB, 'Color', c_orange, 'LineWidth', lw_main);
grid on; box on;
xlabel('Frequency (Hz)');
ylabel('Power / Frequency (dB/Hz)');
title('PSD of noise-corrupted and de-noised EEG');
legend('Noise-corrupted','ANC de-noised','Location','best');
xlim([0 60]);

nexttile;
psd_diff = abs(psd_orig_dB - psd_denoised_dB);
plot(f_orig, psd_diff, 'Color', c_red, 'LineWidth', lw_main);
grid on; box on;
xlabel('Frequency (Hz)');
ylabel('Absolute difference (dB)');
title('Absolute PSD difference');
xlim([0 60]);