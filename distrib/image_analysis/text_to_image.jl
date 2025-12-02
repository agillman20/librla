# text_to_image.jl - Generate an image with text, auto-wrapped to fit
#
# Usage:
#   julia text_to_image.jl                           # Default text
#   julia text_to_image.jl "Your text here"          # Custom text
#   julia text_to_image.jl Your text here            # Unquoted also works
#
# The text is automatically wrapped and font size adjusted to fit the image.
#
# Requires: Plots

using Plots

# Get text from command line or use default
txt = length(ARGS) >= 1 ? join(ARGS, " ") : "Hello, world!"

# Replace \n with actual newlines
txt = replace(txt, "\\n" => "\n")

"""
Wrap text to target width and calculate optimal font size.
"""
function fit_text(txt::String, target_width::Int=40)
    # Split by existing newlines first
    paragraphs = split(txt, "\n")
    wrapped_lines = String[]

    for para in paragraphs
        if length(para) <= target_width
            push!(wrapped_lines, para)
        else
            # Wrap long lines by words
            words = split(para)
            current_line = ""
            for word in words
                if isempty(current_line)
                    current_line = word
                elseif length(current_line) + 1 + length(word) <= target_width
                    current_line = current_line * " " * word
                else
                    push!(wrapped_lines, current_line)
                    current_line = word
                end
            end
            if !isempty(current_line)
                push!(wrapped_lines, current_line)
            end
        end
    end

    wrapped = join(wrapped_lines, "\n")
    num_lines = length(wrapped_lines)
    max_len = maximum(length.(wrapped_lines); init=1)

    # Calculate font size to fit
    font_by_width = 600 * 15 / max(max_len, 1)
    font_by_height = 600 / max(num_lines, 1)
    fontsize = min(font_by_width, font_by_height, 600)

    return wrapped, round(Int, fontsize)
end

wrapped_txt, fontsize = fit_text(txt)

plot(annotations=(0.5, 0.5, text(wrapped_txt, fontsize, :center, :center, "Courier")),
    grid=:none, frame=:none, size=(4000, 1500))
savefig("text_to_image.png")

println("Saved text_to_image.png with text:")
println("\"$txt\"")
println("Font size: $fontsize, Lines: $(length(split(wrapped_txt, '\n')))")
