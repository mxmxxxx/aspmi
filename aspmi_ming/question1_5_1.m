clc; close all; clear;

%% ----------------------------
% 1) Generate original signal
%% ----------------------------
fs = 200;                 % Hz
N  = 1000;
t  = (0:N-1)/fs;

A   = 1.0;
f0  = 1;                  % Hz
phi = pi/3;               % 60 degrees

x = A*sin(2*pi*f0*t + phi);
x = x(:);                 % column vector (safer)

%% ----------------------------
% 2) iAAFT parameters
%% ----------------------------
tol  = 1e-6;
Nmax = 1000;

x_sorted = sort(x);
X_mag    = abs(fft(x));

% initialise surrogate by shuffling
s = x(randperm(N));

prevMSE = inf;

%% ----------------------------
% 3) iAAFT loop
%% ----------------------------
for it = 1:Nmax
    % Spectral matching: keep current phase, enforce target magnitudes
    S  = fft(s);
    S2 = X_mag .* exp(1j*angle(S));
    s2 = real(ifft(S2));

    % Amplitude matching: enforce target amplitude distribution (rank-order)
    [~, idx] = sort(s2);
    s(idx) = x_sorted;

    % Convergence check on magnitude spectrum
    mse = mean((abs(fft(s)) - X_mag).^2);
    if abs(mse - prevMSE) < tol
        break;
    end
    prevMSE = mse;
end

fprintf('iAAFT finished in %d iterations, final MSE = %.3e\n', it, mse);

%% ----------------------------
% 4) One-sided magnitude spectra
%% ----------------------------
Nfft = N; % keep simple
X1 = fft(x, Nfft);
S1 = fft(s, Nfft);

% one-sided indices (include DC and Nyquist if even)
K = floor(Nfft/2) + 1;
f = (0:K-1)*(fs/Nfft);

Xmag = abs(X1(1:K));
Smag = abs(S1(1:K));

%% ----------------------------
% 5) Amplitude distributions (legacy-safe)
%% ----------------------------
nbins = 30;

% use COMMON bin centers so the overlay is meaningful
xmin = min([x; s]);
xmax = max([x; s]);
edges = linspace(xmin, xmax, nbins);   % these are bin centers for hist()

[count_x, centers] = hist(x, edges);
[count_s, ~]       = hist(s, edges);

binw = centers(2) - centers(1);
pdf_x = count_x / (sum(count_x)*binw);
pdf_s = count_s / (sum(count_s)*binw);

%% ----------------------------
% 6) Plot 2x2 figure (no fancy features)
%% ----------------------------
figure(1); clf;

% (a) Original time domain (show first 1s to make it readable)
subplot(2,2,1)
plot(t, x, 'LineWidth', 1.2);
grid on
xlim([0 1])
xlabel('Time (s)'); ylabel('Amplitude');
title('Original signal');

% (b) Surrogate time domain (first 1s)
subplot(2,2,2)
plot(t, s, 'LineWidth', 1.2);
grid on
xlim([0 1])
xlabel('Time (s)'); ylabel('Amplitude');
title('iAAFT surrogate');

% (c) Magnitude spectrum overlay
subplot(2,2,3)
plot(f, Xmag, 'LineWidth', 1.2); hold on
plot(f, Smag, '--', 'LineWidth', 1.2);
grid on
xlim([0 25])
xlabel('Frequency (Hz)'); ylabel('Magnitude');
title('Magnitude spectrum');
legend('Original','Surrogate');

% (d) Amplitude distribution overlay (line plot = most compatible)
subplot(2,2,4)
plot(centers, pdf_x, 'LineWidth', 1.2); hold on
plot(centers, pdf_s, '--', 'LineWidth', 1.2);
grid on
xlabel('Amplitude'); ylabel('Probability density');
title('Amplitude distribution');
legend('Original','Surrogate');

drawnow;
