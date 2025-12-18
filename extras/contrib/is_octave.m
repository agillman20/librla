function r = is_octave()

r = exist('OCTAVE_VERSION', 'builtin') ~= 0;
