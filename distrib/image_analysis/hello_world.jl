# https://tobydriscoll.net/fnc-julia/matrixanaly/dimreduce.html
# Copyright(C) 2023, By Tobin A. Driscoll and Richard J. Braun
    
using Plots, Images, LinearAlgebra

plot(annotations=(0.5,0.5,text("Hello, world!",400,:center,:center)),
    grid=:none,frame=:none,size=(4000,1500))
savefig("hello_world.png")
img = load("hello.png")
A = @. Float64(Gray(img))
Gray.(A)

U,σ,V = svd(A)
scatter(σ,xaxis=("i"), yaxis=(:log10,"sigma_i"), title="Singular values")
