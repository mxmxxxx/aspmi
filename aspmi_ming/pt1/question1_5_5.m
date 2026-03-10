clc; close all; clear;

%% ============================
% Parameters + non-stationary signal
%% ============================
fs = 256;
T  = 8;
N  = fs*T;
t  = (0:N-1)'/fs;

% Sigmoid envelopes (exact forms from spec)
A_delta = 5 - 4./(1 + exp(-5*(t-2)));
A_alpha = 1 + 4./(1 + exp(-5*(t-6)));

% Components + noise
sigma = 0.8;
eta = sigma*randn(size(t));

x = A_delta .* cos(2*pi*2*t) + A_alpha .* cos(2*pi*10*t) + eta;

%% ============================
% (a) Stationary iAAFT surrogate on full record
%% ============================
x_surr = iAAFT_legacy(x, 1e-6, 1000);

% FFT magnitude (one-sided)
[f_fft, Xmag] = oneSidedMag(x, fs);
[~,    Smag] = oneSidedMag(x_surr, fs);

% Spectrogram settings
winLen = 128;
nover  = winLen/2;
nfft   = 256;

% Use spectrogram if available; otherwise use specgram (older MATLAB)
[Sx,Fx,Tx] = spectrogram(x, winLen, nover, nfft, fs);
[Ss,Fs,Ts] = spectrogram(x_surr, winLen, nover, nfft, fs);

Sx_dB = 20*log10(abs(Sx) + 1e-12);
Ss_dB = 20*log10(abs(Ss) + 1e-12);

%% 2x3 Figure: original vs stationary surrogate
figure(1); clf;

% Row 1: original
subplot(2,3,1)
plot(t, x, 'LineWidth', 1.0); grid on
xlabel('Time (s)'); ylabel('Amplitude');
title('Original: time domain')

subplot(2,3,2)
plot(f_fft, Xmag, 'LineWidth', 1.0); grid on
xlim([0 40])
xlabel('Frequency (Hz)'); ylabel('Magnitude');
title('Original: |FFT|')

subplot(2,3,3)
imagesc(Tx, Fx, Sx_dB); axis xy; colorbar
ylim([0 40])
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Original: spectrogram (STFT)')

% Row 2: surrogate
subplot(2,3,4)
plot(t, x_surr, 'LineWidth', 1.0); grid on
xlabel('Time (s)'); ylabel('Amplitude');
title('Stationary iAAFT: time domain')

subplot(2,3,5)
plot(f_fft, Smag, 'LineWidth', 1.0); grid on
xlim([0 40])
xlabel('Frequency (Hz)'); ylabel('Magnitude');
title('Stationary iAAFT: |FFT|')

subplot(2,3,6)
imagesc(Ts, Fs, Ss_dB); axis xy; colorbar
ylim([0 40])
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Stationary iAAFT: spectrogram (STFT)')

%% ============================
% (c) Segmentation-based iAAFT for non-stationarity
%% ============================
% Changepoint: dominance switches around the midpoint (~4 s).
cp = round(4*fs);   % sample index for 4 seconds

x1 = x(1:cp);
x2 = x(cp+1:end);

s1 = iAAFT_legacy(x1, 1e-6, 1000);
s2 = iAAFT_legacy(x2, 1e-6, 1000);

% Concatenate with optional crossfade to reduce boundary artefacts
useCrossfade = true;
fadeLen = round(0.2*fs); % 200 ms crossfade

if useCrossfade && fadeLen > 1 && fadeLen < min(length(s1), length(s2))
    w = linspace(0,1,fadeLen)';              % fade-in
    s2(1:fadeLen) = (1-w).*s1(end-fadeLen+1:end) + w.*s2(1:fadeLen);
end

x_ns_surr = [s1; s2];

% Spectrogram for segmented surrogate
[Sn,Fn,Tn] = spectrogram(x_ns_surr, winLen, nover, nfft, fs);
Sn_dB = 20*log10(abs(Sn) + 1e-12);

%% Plot segmented surrogate comparison (optional figure)
figure(2); clf;

subplot(2,1,1)
plot(t, x, 'LineWidth', 1.0); hold on
xline(4,'--','Changepoint','LineWidth',1.0);
grid on
xlabel('Time (s)'); ylabel('Amplitude');
title('Original non-stationary signal (changepoint at 4 s)')

subplot(2,1,2)
imagesc(Tn, Fn, Sn_dB); axis xy; colorbar
ylim([0 40])
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Segmented iAAFT surrogate: spectrogram (STFT)')

%% ============================
% Local helper functions (legacy-safe)
%% ============================
function s = iAAFT_legacy(x, tol, Nmax)
    x = x(:);
    N = length(x);
    x_sorted = sort(x);
    X_mag = abs(fft(x));

    s = x(randperm(N));
    prevMSE = inf;

    for it = 1:Nmax
        S  = fft(s);
        S2 = X_mag .* exp(1j*angle(S));
        s2 = real(ifft(S2));

        [~, idx] = sort(s2);
        s(idx) = x_sorted;

        mse = mean((abs(fft(s)) - X_mag).^2);
        if abs(mse - prevMSE) < tol
            break;
        end
        prevMSE = mse;
    end
end

function [f, mag] = oneSidedMag(x, fs)
    x = x(:);
    N = length(x);
    X = fft(x);
    K = floor(N/2) + 1;
    f = (0:K-1)'*(fs/N);
    mag = abs(X(1:K));
end
