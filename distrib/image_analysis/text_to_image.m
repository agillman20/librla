function text_to_image(varargin)
%TEXT_TO_IMAGE  Generate an image with text, auto-wrapped to fit.
%
%  Usage:
%    text_to_image                          % Default text
%    text_to_image('Your text here')        % Custom text
%    octave text_to_image.m Your text here  % From command line
%
%  The text is automatically wrapped and font size adjusted to fit the image.

% Get text from command line (Octave argv) or function arguments (varargin)
if exist('argv', 'builtin') && ~isempty(argv())
    % Octave command line: octave text_to_image.m Your text
    txt = strrep(strjoin(argv(), ' '), '\n', char(10));
elseif nargin >= 1
    % Function call (MATLAB or Octave): text_to_image('Your text')
    txt = strrep(strjoin(varargin, ' '), '\n', char(10));
else
    txt = 'Hello, world!';
end

% Image dimensions
img_width = 4000;
img_height = 1500;

% Wrap text and calculate font size
target_chars = 40;
[wrapped_txt, fontsize] = fit_text(txt, target_chars);

fig = figure('Position', [100 100 img_width img_height], 'Color', 'w');
axes('Position', [0 0 1 1]);
text(0.5, 0.5, wrapped_txt, 'FontSize', fontsize, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontName', 'FixedWidth', 'Interpreter', 'none');
axis off
xlim([0 1])
ylim([0 1])

print(fig, 'text_to_image.png', '-dpng', '-r100');
fprintf('Saved text_to_image.png with text:\n"%s"\n', txt);
fprintf('Font size: %.0f, Lines: %d\n', fontsize, numel(strfind(wrapped_txt, char(10)))+1);
close(fig);

end

function [wrapped, fontsize] = fit_text(txt, target_width)
    % Wrap text to target width and calculate optimal font size

    % Split by existing newlines first
    lines = strsplit(txt, char(10));
    wrapped_lines = {};

    for i = 1:length(lines)
        line = lines{i};
        if length(line) <= target_width
            wrapped_lines{end+1} = line;
        else
            % Wrap long lines
            words = strsplit(line);
            current_line = '';
            for j = 1:length(words)
                word = words{j};
                if isempty(current_line)
                    current_line = word;
                elseif length(current_line) + 1 + length(word) <= target_width
                    current_line = [current_line ' ' word];
                else
                    wrapped_lines{end+1} = current_line;
                    current_line = word;
                end
            end
            if ~isempty(current_line)
                wrapped_lines{end+1} = current_line;
            end
        end
    end

    wrapped = strjoin(wrapped_lines, char(10));
    num_lines = length(wrapped_lines);
    max_len = max(cellfun(@length, wrapped_lines));

    % Calculate font size to fit (tuned for 4000x1500 image)
    font_by_width = 550 * 15 / max(max_len, 1);
    font_by_height = 550 / max(num_lines, 1);
    fontsize = min([font_by_width, font_by_height, 500]);
end
