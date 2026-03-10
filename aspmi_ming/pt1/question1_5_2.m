clc; close all; clear;

%% ----------------------------
% Signal definition
%% ----------------------------
fs = 200;        % Hz
N  = 200;
n  = (0:N-1)';

x = cos(2*pi*8*n/fs) + 0.5*cos(2*pi*20*n/fs);

%% ----------------------------
% iAAFT surrogate (legacy-safe)
%% ----------------------------
tol  = 1e-6;
Nmax = 1000;

x_sorted = sort(x);
X_mag    = abs(fft(x));

s = x(randperm(N));   % random shuffle init
prevMSE = inf;

for it = 1:Nmax
    % spectral match
    S  = fft(s);
    S2 = X_mag .* exp(1j*angle(S));
    s2 = real(ifft(S2));

    % amplitude match
    [~, idx] = sort(s2);
    s(idx) = x_sorted;

    % convergence on magnitude spectrum
    mse = mean((abs(fft(s)) - X_mag).^2);
    if abs(mse - prevMSE) < tol
        break;
    end
    prevMSE = mse;
end

fprintf('iAAFT finished in %d iterations, final MSE = %.3e\n', it, mse);

%% ----------------------------
% Plot time-domain (original vs surrogate)
%% ----------------------------
t = n/fs;

figure(1); clf;
plot(t, x, 'LineWidth', 1.2); hold on;
plot(t, s, '--', 'LineWidth', 1.2);
grid on
xlabel('Time (s)'); ylabel('Amplitude');
title('Original signal vs iAAFT surrogate (time domain)');
legend('Original','iAAFT surrogate');

%% ----------------------------
% Autocovariance r[k], k=0..100
%% ----------------------------
Kmax = 100;

% Remove mean (autocovariance)
x0 = x - mean(x);
s0 = s - mean(s);

r_x = zeros(Kmax+1,1);
r_s = zeros(Kmax+1,1);

for k = 0:Kmax
    r_x(k+1) = (1/(N-k)) * sum( x0(1+ k:end) .* conj(x0(1:end-k)) );
    r_s(k+1) = (1/(N-k)) * sum( s0(1+ k:end) .* conj(s0(1:end-k)) );
end

figure(2); clf;
stem(0:Kmax, r_x, 'filled'); hold on;
stem(0:Kmax, r_s, 'r');
grid on
xlabel('Lag k'); ylabel('Autocovariance r[k]');
title('Autocovariance comparison: original vs iAAFT surrogate');
legend('Original','iAAFT surrogate');
