clc; close all; clear;

%% ----------------------------
% Signal setup
%% ----------------------------
fs = 100;           % Hz
T  = 10;            % seconds
N  = fs*T;          % samples (1000)
t  = (0:N-1)'/fs;

f1 = 1.2;           % Hz (bin-centered for df=0.1)
f2 = 1.25;          % Hz (between bins)

x1 = sin(2*pi*f1*t);
x2 = sin(2*pi*f2*t);

%% ----------------------------
% iAAFT surrogates (legacy-safe)
%% ----------------------------
s1 = iAAFT_legacy(x1, 1e-6, 1000);
s2 = iAAFT_legacy(x2, 1e-6, 1000);

%% ----------------------------
% Time-domain plots
%% ----------------------------
figure(1); clf;

subplot(2,2,1)
plot(t, x1, 'LineWidth', 1.2); grid on
xlim([0 10])
xlabel('Time (s)'); ylabel('Amplitude');
title('Original: 1.2 Hz')

subplot(2,2,2)
plot(t, s1, 'LineWidth', 1.2); grid on
xlim([0 10])
xlabel('Time (s)'); ylabel('Amplitude');
title('iAAFT surrogate: 1.2 Hz')

subplot(2,2,3)
plot(t, x2, 'LineWidth', 1.2); grid on
xlim([0 10])
xlabel('Time (s)'); ylabel('Amplitude');
title('Original: 1.25 Hz')

subplot(2,2,4)
plot(t, s2, 'LineWidth', 1.2); grid on
xlim([0 10])
xlabel('Time (s)'); ylabel('Amplitude');
title('iAAFT surrogate: 1.25 Hz (beating may appear)')

%% ----------------------------
% Magnitude spectra (stem, zoom 0.5–2.0 Hz)
%% ----------------------------
% Use standard FFT grid (no zero-padding) to highlight bin alignment
X1 = fft(x1);
X2 = fft(x2);
S1 = fft(s1);
S2 = fft(s2);

% One-sided frequency axis
K = floor(N/2) + 1;
f = (0:K-1)'*(fs/N);

magX1 = abs(X1(1:K));
magS1 = abs(S1(1:K));
magX2 = abs(X2(1:K));
magS2 = abs(S2(1:K));

% Zoom indices
idx = find(f >= 0.5 & f <= 2.0);

figure(2); clf;

subplot(2,1,1)
stem(f(idx), magX1(idx), 'filled'); hold on
stem(f(idx), magS1(idx));
grid on
xlabel('Frequency (Hz)'); ylabel('Magnitude');
title('Magnitude spectrum (stem): 1.2 Hz (bin-aligned)')
legend('Original','Surrogate','Location','best')

subplot(2,1,2)
stem(f(idx), magX2(idx), 'filled'); hold on
stem(f(idx), magS2(idx));
grid on
xlabel('Frequency (Hz)'); ylabel('Magnitude');
title('Magnitude spectrum (stem): 1.25 Hz (between bins)')
legend('Original','Surrogate','Location','best')

%% ============================
% Local function: iAAFT (legacy-safe)
%% ============================
function s = iAAFT_legacy(x, tol, Nmax)
    x = x(:);
    N = length(x);

    x_sorted = sort(x);
    X_mag    = abs(fft(x));

    % initial surrogate
    s = x(randperm(N));
    prevMSE = inf;

    for it = 1:Nmax
        % spectral match
        S  = fft(s);
        S2 = X_mag .* exp(1j*angle(S));
        s2 = real(ifft(S2));

        % amplitude match
        [~, idx] = sort(s2);
        s(idx) = x_sorted;

        % convergence
        mse = mean((abs(fft(s)) - X_mag).^2);
        if abs(mse - prevMSE) < tol
            break;
        end
        prevMSE = mse;
    end
end
