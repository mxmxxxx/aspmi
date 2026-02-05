close all
clear all
clc

%% Generation of complex-valued signal with complex noise
fs = 1;
N = 30;
n = 0:N-1;
nFFT = 256;
n_iter = 500;
Pseudospectrum = zeros(n_iter,nFFT);
m = (N/2)-1;

% Colour palette
c_real   = [0.60 0.85 0.90];   % light cyan (individual realisations)
c_mean   = [0.00 0.45 0.74];   % dark blue (mean)
c_std    = [0.85 0.33 0.10];   % orange (std)

lw_mean = 2.2;
lw_std  = 1.6;

figure(1);

%% -------------------------------
% Left subplot: MUSIC pseudospectrum
%% -------------------------------
subplot(1,2,1)
hold on

for i = 1:n_iter
    
    noise = 0.05/sqrt(2) * (randn(size(n)) + 1j*randn(size(n)));
    x = exp(1j*2*pi*0.3*n) + exp(1j*2*pi*0.32*n) + noise;

    % MUSIC method
    [X,R] = corrmtx(x,m,'modified');
    [S,F] = pmusic(R,2,nFFT,fs,'corr');
    
    Pseudospectrum(i,:) = S;

    % Plot individual realisation (light colour)
    plot(F, S, 'Color', c_real, 'LineWidth', 0.8);
end

% Plot mean pseudospectrum
avg_val = mean(Pseudospectrum);
plot(F, avg_val, 'Color', c_mean, 'LineWidth', lw_mean);

grid on
xlim([0.25 0.40])
xlabel('Normalised frequency (Hz)','FontSize',11)
ylabel('Pseudospectrum','FontSize',11)
title('PSD estimate of signal using MUSIC','FontSize',11)

%% -------------------------------
% Right subplot: standard deviation
%% -------------------------------
subplot(1,2,2)

std_val = std(Pseudospectrum);

plot(F, std_val, 'Color', c_std, 'LineWidth', lw_std);
grid on
xlim([0.25 0.40])
xlabel('Normalised frequency (Hz)','FontSize',11)
ylabel('Pseudospectrum standard deviation','FontSize',11)
title('Standard deviation of MUSIC PSD estimate','FontSize',11)
