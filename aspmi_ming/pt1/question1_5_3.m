clc; close all;

%% Parameters
fs = 200;
N  = length(x);

% FFTs
X = fft(x);
S = fft(s);

% Frequency axis
f = (0:N-1)*(fs/N);

% Find indices closest to 8 Hz and 20 Hz
[~, idx8]  = min(abs(f - 8));
[~, idx20] = min(abs(f - 20));

% Extract phases
phase_x  = [angle(X(idx8)),  angle(X(idx20))];
phase_s  = [angle(S(idx8)),  angle(S(idx20))];

%% Polar plots
figure;

% Original signal phases
subplot(1,2,1)
polarplot([0 phase_x(1)], [0 1], 'LineWidth', 2); hold on
polarplot([0 phase_x(2)], [0 1], 'LineWidth', 2);
title('Original signal phase')
legend('8 Hz','20 Hz','Location','bestoutside')

% Surrogate signal phases
subplot(1,2,2)
polarplot([0 phase_s(1)], [0 1], 'LineWidth', 2); hold on
polarplot([0 phase_s(2)], [0 1], 'LineWidth', 2);
title('iAAFT surrogate phase')
legend('8 Hz','20 Hz','Location','bestoutside')
