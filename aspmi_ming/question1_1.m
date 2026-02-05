close all
clear all
clc

%% =========================
%  ACF + PSD comparison (WGN vs Sinusoid)
%  Colour-customised full script
%% =========================

%% --- Colour palette (colour-blind safe, report friendly)
c_blue   = [0.00 0.45 0.74];
c_orange = [0.85 0.33 0.10];
c_green  = [0.47 0.67 0.19];
c_red    = [0.64 0.08 0.18];
c_purple = [0.49 0.18 0.56];
c_gray   = [0.50 0.50 0.50];

%% --- Global plot defaults (optional, makes everything consistent)
set(groot,'defaultAxesFontSize',11)
set(groot,'defaultLineLineWidth',1.4)

%% Definition of signals
L = 10000;                 % Signal length
noise = randn(L,1);        % White Gaussian noise generated

time_lag1 = (-(L-1):L-1);
index = find(time_lag1==0);
acf1 = zeros(1,(2*L-1));
acf1(index) = 1;           % Variance of the noise (set to 1 here)

%% --- WGN ACF plot (left panel of Figure 1)
figure(1);
subplot(1,2,1)
plot(time_lag1, acf1, 'Color', c_blue, 'LineWidth', 1.5);
grid on
xlabel('Time lag (sample)')
ylabel('ACF')
title({'Theoretical ACF','of White Gaussian Noise'})

%% --- Define sinusoid and its theoretical ACF
T = (0:L-1);
sine_wave = sin(T./100);

% Theoretical ACF for random-phase sinusoid: r(k) = (1/2)cos(w0 k)
% Here w0 = 1/100 (rad/sample) since sine_wave = sin(n/100)
acf2 = (1/2).*cos(time_lag1./100);
time_lag2 = time_lag1;

%% --- Sinusoid ACF plot (left panel of Figure 2)
figure(2);
subplot(1,2,1)
plot(time_lag2, acf2, 'Color', c_orange, 'LineWidth', 1.5);
grid on
xlabel('Time lag (sample)')
ylabel('ACF')
title({'Theoretical ACF','of Sinusoidal Signal'})

%% =========================
%  Definition 1 of PSD (DTFT of ACF)
%  Steps:
%   1) ifftshift -> move lag-0 to start
%   2) fft -> DTFT approximation
%   3) fftshift -> center DC
%% =========================
psd1 = fftshift(fft(ifftshift(acf1)));
psd2 = fftshift(fft(ifftshift(acf2)));

fs = 1;                      % Sampling frequency for normalisation
n1 = length(psd1);

% Frequency axis (cycles/sample). If you want "×π rad/sample" labelling,
% keep as-is (this is what your original code did).
freqAxis1 = (-n1/2:n1/2-1) * (fs/n1);

%% =========================
%  Definition 2 of PSD (Expected periodogram)
%  Here you are using a single realisation (so it's a periodogram estimate).
%% =========================
psd_estimate_noise1 = fftshift(fft(noise, n1));
psd_estimate_noise1 = (abs(psd_estimate_noise1).^2) ./ L;

psd_estimate_2 = fftshift(fft(sine_wave, n1));
psd_estimate_2 = (abs(psd_estimate_2).^2) ./ L;

n2 = length(psd_estimate_noise1);
freqAxis2 = (-n2/2:n2/2-1) * (fs/n2);

%% =========================
%  Comparison plots
%% =========================

%% --- Figure 1: WGN PSD comparison (right panel)
figure(1);
subplot(1,2,2)

% Definition 2 (periodogram) - grey noisy curve
plot(freqAxis2, psd_estimate_noise1, ...
     'Color', c_gray, 'LineWidth', 0.8); hold on

% Mean level of Definition 2 - green dashed line
yline(mean(psd_estimate_noise1), '--', ...
      'Color', c_green, 'LineWidth', 1.6);

% Definition 1 (DTFT of ACF) - blue solid
plot(freqAxis1, real(psd1), ...
     'Color', c_blue, 'LineWidth', 1.6);

grid on
title({'Periodogram of WGN using','the two definitions of PSD'})
ylabel('PSD')
xlabel('Normalised frequency (\pi rad/sample)')

legend('Definition 2 (Periodogram)', ...
       'Mean of Definition 2', ...
       'Definition 1 (DTFT of ACF)', ...
       'FontSize',9, 'Location','best')

%% --- Figure 2: Sinusoid PSD comparison (right panel)
figure(2);
subplot(1,2,2)

% Definition 1 (DTFT of ACF) - blue solid
plot(freqAxis1, real(psd2), ...
     'Color', c_blue, 'LineWidth', 1.6); hold on

% Definition 2 (periodogram) - red
plot(freqAxis2, psd_estimate_2, ...
     'Color', c_red, 'LineWidth', 1.0);

grid on
title({'Periodogram of sinusoidal signal','using the two definitions of PSD'})
ylabel('PSD')
xlabel('Normalised frequency (\pi rad/sample)')

legend('Definition 1 (DTFT of ACF)', ...
       'Definition 2 (Periodogram)', ...
       'FontSize',10, 'Location','best')

%% --- Optional: if you want a quick look at the sine wave itself
% figure(3);
% plot(sine_wave, 'Color', c_purple);
% grid on; title('Sinusoidal signal'); xlabel('n'); ylabel('Amplitude');
 