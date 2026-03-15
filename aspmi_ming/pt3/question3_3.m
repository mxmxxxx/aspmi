%% Assignment 3.3 - A Real-Time Spectrum Analyser Using LMS
close all; clear; clc;

%% ========================================================================
%% Global plotting style
%% ========================================================================
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

rng('default');

%% ========================================================================
%% Part (a) - DFT-CLMS on frequency-modulated signal
%% ========================================================================
N  = 1500;      % Number of samples
fs = 2000;      % Sampling frequency
sigma2_eta = 0.05;

n = 1:N;

% Circular complex white Gaussian noise
eta = sqrt(sigma2_eta) * randn(1,N) + 1j * sqrt(sigma2_eta) * randn(1,N);

% Instantaneous frequency
f_n = [100*ones(1,500), ...
       100 + ((501:1000)-500)/2, ...
       100 + (((1001:1500)-1000)/25).^2];

% Phase by integration
phi = cumtrapz(f_n);

% FM signal
y = exp(1j*(2*pi/fs)*phi) + eta;

% DFT-CLMS parameters
L = 1024;
mu = 1;
gamma_values = [0, 0.01, 0.1, 0.5];

% DFT basis matrix
x = (1/L) * exp(1j * 2 * pi * (0:N-1)' * (0:L-1) / L).';

%% Plot time-frequency results for different gamma
figure('Name','DFT-CLMS on FM Signal', ...
       'Color','w','Position',[60 60 1300 900]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

for k = 1:length(gamma_values)
    gamma = gamma_values(k);

    % Run DFT-CLMS
    [a_hat, e] = clms_dft(x, y, mu, gamma, L); %#ok<NASGU>

    H = abs(a_hat);

    % Clip large outliers for display
    clip_level = 50 * median(H(:));
    H(H > clip_level) = clip_level;

    nexttile;
    surf(1:N, (0:L-1)*(fs/L), H, 'LineStyle', 'none');
    view(2);
    axis tight;
    ylim([0 700]);
    colormap turbo;
    colorbar('TickLabelInterpreter','latex');
    xlabel('Time index $n$');
    ylabel('Frequency (Hz)');
    title(['DFT-CLMS, $\gamma = ', num2str(gamma), '$, $\mu = 1$']);
end

sgtitle('Time-frequency estimation of the FM signal using DFT-CLMS', ...
        'Interpreter','latex','FontSize',18);

%% ========================================================================
%% Part (b) - DFT-CLMS on EEG segment
%% ========================================================================
load('EEG_Data_Assignment1.mat');

% Select EEG segment
N  = 1200;
fs = 1200;
start_idx = 1000;

y = POz(start_idx:start_idx+N-1);
y = y - mean(y);     % Remove DC component

% DFT-CLMS parameters
L = 1024;
mu = 1;
gamma_values = [0, 0.1];

% DFT basis matrix
x = (1/L) * exp(1j * 2 * pi * (0:N-1)' * (0:L-1) / L).';

%% Plot EEG time-frequency estimates
figure('Name','DFT-CLMS on EEG Segment', ...
       'Color','w','Position',[100 100 1200 450]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

for k = 1:length(gamma_values)
    gamma = gamma_values(k);

    % Run DFT-CLMS
    [a_hat, e] = clms_dft(x, y, mu, gamma, L); %#ok<NASGU>

    H = abs(a_hat);

    % Clip large outliers for display
    clip_level = 50 * median(H(:));
    H(H > clip_level) = clip_level;

    nexttile;
    surf(1:N, (0:L-1)*(fs/L), H, 'LineStyle', 'none');
    view(2);
    axis tight;
    ylim([0 100]);
    colormap turbo;
    colorbar('TickLabelInterpreter','latex');
    xlabel('Time index $n$');
    ylabel('Frequency (Hz)');
    title(['EEG DFT-CLMS, $\gamma = ', num2str(gamma), '$, $\mu = 1$']);
end

sgtitle('Time-frequency estimation of EEG using DFT-CLMS', ...
        'Interpreter','latex','FontSize',18);