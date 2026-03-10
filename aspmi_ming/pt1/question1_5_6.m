clc; close all; clear;

%% ============================
% Folder + files  (same logic as your code)
%% ============================
dataFolder = fullfile(pwd, 'aspmi_ming', 'brainwave_samples');

files = {'alpha_waves_data.mat','delta_waves_data.mat','Kcomplex_data.mat', ...
         'spindles_data.mat','theta_waves_data.mat'};

fs = 100;            % given in coursework
tol  = 1e-6;         % iAAFT convergence
Nmax = 1000;
maxLagSec = 2;       % cross-corr plot range (+/- seconds)

for fi = 1:length(files)
    fname = fullfile(dataFolder, files{fi});
    if ~exist(fname,'file')
        warning('File not found: %s (skipping)', fname);
        continue;
    end

    %% --------- Load .mat ----------
    D = load(fname);

    % Try to read variable called "data" first, otherwise pick first numeric array
    if isfield(D,'data')
        A = D.data;
    else
        fn = fieldnames(D);
        A = [];
        for k = 1:length(fn)
            v = D.(fn{k});
            if isnumeric(v) && ~isempty(v)
                A = v;
                break;
            end
        end
        if isempty(A)
            error('No numeric array found inside %s', files{fi});
        end
    end

    A = double(A);

    %% --------- Extract two channels ----------
    % Accept N×2 or 2×N
    if ndims(A) ~= 2
        error('Unexpected array dimensions in %s. Expected 2-D array.', files{fi});
    end

    if size(A,2) == 2
        x = A(:,1);
        y = A(:,2);
    elseif size(A,1) == 2
        x = A(1,:).';
        y = A(2,:).';
    else
        error('Could not interpret %s as two-channel EEG. Size is %dx%d.', ...
              files{fi}, size(A,1), size(A,2));
    end

    % Match length, demean
    N = min(length(x), length(y));
    x = x(1:N) - mean(x(1:N));
    y = y(1:N) - mean(y(1:N));

    %% --------- Generate iAAFT surrogates ----------
    xs = iAAFT_legacy(x, tol, Nmax);
    ys = iAAFT_legacy(y, tol, Nmax);

    %% --------- Time-domain plots (first 10 s) ----------
    t = (0:N-1)'/fs;
    tShow = min(10, t(end));
    idxShow = find(t <= tShow);

    figure('Color','w'); clf;
    subplot(2,1,1)
    plot(t(idxShow), x(idxShow), 'LineWidth', 1.0); hold on;
    plot(t(idxShow), xs(idxShow), '--', 'LineWidth', 1.0);
    grid on
    xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('%s | Channel 1 (Original vs iAAFT)', files{fi}), 'Interpreter','none');
    legend('Original','Surrogate','Location','best');

    subplot(2,1,2)
    plot(t(idxShow), y(idxShow), 'LineWidth', 1.0); hold on;
    plot(t(idxShow), ys(idxShow), '--', 'LineWidth', 1.0);
    grid on
    xlabel('Time (s)'); ylabel('Amplitude');
    title('Channel 2 (Original vs iAAFT)');
    legend('Original','Surrogate','Location','best');

    %% --------- Cross-correlation ----------
    maxLag = round(maxLagSec*fs);
    [r_xy, lags] = xcorr(x, y, maxLag, 'coeff');
    [r_s , ~   ] = xcorr(xs, ys, maxLag, 'coeff');
    lagSec = lags/fs;

    figure('Color','w'); clf;
    plot(lagSec, r_xy, 'LineWidth', 1.2); hold on;
    plot(lagSec, r_s, '--', 'LineWidth', 1.2);
    grid on
    xlabel('Lag (s)'); ylabel('Cross-correlation (coeff)');
    title(sprintf('Cross-correlation: original vs surrogate | %s', files{fi}), 'Interpreter','none');
    legend('Original channels','Surrogate channels','Location','best');
end

%% ============================
% iAAFT (legacy-safe)
%% ============================
function s = iAAFT_legacy(x, tol, Nmax)
    x = x(:);

    x_sorted = sort(x);
    X_mag    = abs(fft(x));

    s = x(randperm(length(x)));
    prevMSE = inf;

    for it = 1:Nmax
        % spectral matching
        S  = fft(s);
        S2 = X_mag .* exp(1j*angle(S));
        s2 = real(ifft(S2));

        % amplitude matching
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
