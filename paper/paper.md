---
title: 'librla: A library of randomized linear algebra routines'
tags:
  - Python
  - Julia
  - Matlab
  - QR, SVD and Interpolatory decomposition
authors:
  - name: Adrianna Gillman
    orcid: 0000-0002-3413-4133
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Zydrunas Gimbutas
    orcid: 0000-0003-3209-8228
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: University of Colorado, Boulder, Department of Applied Mathematics
   index: 1
 - name: National Institute of Standards and Technology (NIST)
   index: 2
 
date: March 18, 2026
bibliography: paper.bib

# Optional fields if submitting to an AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Journal of Open Source Software 
#<- The name of the AAS journal.
---

# Summary

Randomized linear algebra algorithms have become a vital tool for a variety of areas including fast direct solvers, reduced order modeling, and data science.   Additionally, randomized linear algebra provides useful routines for solving total least squares problems and rank-deficient least squares problems, doing matrix approximation, and skeletonizing (i.e. subset selection) a matrix [@Chan:1992].  \texttt{librla}, written natively in Python, Julia and Matlab, provides low-rank QR factorizations, SVDs and interpolatory decompositions.  The algorithms randomly sample the range of the matrix or operator in a similar manner to [@Halko:2011]. A key feature of this package is that it is designed to exploit Level 3 BLAS operators as much as possible.  \texttt{librla} is designed for small to mid-range sized matrices (i.e. up to roughly 10,000 in size depending on computing resources).  \texttt{librla} is not intended for matrices that are larger or do not fit in memory.

\texttt{librla} includes the following options, all of which can be used for both real and complex matrices: 

- \texttt{qr\_sketch}: Randomized QR factorization.
- \texttt{svd\_sketch}: Randomized Singular Value Decomposition (SVD).
- \texttt{id\_sketch}: Interpolatory Decomposition via randomized sampling.

The user has the choice of specifying a tolerance or a desired rank.  For each method, the user has the option to use extra samples and to use a specified number of power iterations.  The package does include the option to create low-rank factorizations of matrices that are applied via matrix-vector multiplication codes. In order to use this option, \texttt{librla} requires the ability to apply both the matrix and its transpose via a subroutine.

While the algorithm used in \texttt{librla} is built in a similar manner to the randomized methods in [@Halko:2011;@Liberty:2007]. There is a large collection of related work.  [@Martinsson:2020;
@Mahoney:2011;@Drineas2017LecturesOR;@Woodruff:2014;@Voronin:2017; ] are some review papers on the field.  Some literature [@Mahoney:2009;@Sorensen:2016; @Drineas:2008] has focused on the design of CUR factorizations .  While other manuscripts [@Frieze:2004; @Drineas:2006_2; @Drinea:2006_3] have focused on monte carlo approaches to creating the low rank factorizations.  Randomized projection methods [@Halko:2011;@Liberty:2007;@Duersch:2020;@MEIER:2024; @Gu:1996; @ERICHSON:2018;@Rokhlin:2008] are common.  Deterministic factorization techniques exist [@Duersch:2017_2;@Feng:2018].  Many techniques are specifically designed for distributed memory  and/or to not require many passes through the data [;@Martinsson:2019;@Sarlos:2006; @Tropp:2017;@Yu:2017;@Lindquist:2020;@Yang:2015;@Bjarkason:2019;].  There is much literature from the randomized linear algebra community for applications including $L_2$ solutions [@Drineas:2006;@Drineas:2007;@Avron:2010; @Meng:2014;@Pilanci:2016], reduced order modelling [@Sorensen:2016], linear programming [@Drineas:2012;@Chowdhury:2020;], statisics [@Drineas:2016], machine learning [@Yao:2021],
Newton methods [@Roosta-Khorasani:2019; @Pilanci:2017], and differentially private matrices [@Balu:2016; @Cormode:2018; @Choi:2020].
The user-specified rank algorithm in \texttt{librla} is influenced by the \texttt{svd\_lowrank} routine in PyTorch.


# Statement of need

While there is a large amount of research activity in the field of randomized linear algebra, an easy-to-use, efficient and stable factorization library is not available.  For several factorization libraries, the stability issue manafest itself by either failure in the form of an invalid factorization or being unable to return anything.  When new techniques implemented in a high level language are depdent on low level language libraries, it has a big impact on the portability and usability of the new work.  This problem has had an impact of the field of fast direct solvers for boundary integral equations which often depend on a legacy Fortran library [Tygert:2008].  

Currently, there are two related routines in widely available, maintained Python packages.  They are PyTorch's \texttt{svd\_lowrank} and SciPy's \texttt{id\_decomp} found in the \texttt{scipy.linalg.interpolative} library.  Codes allowing the user to compare the performance are available in the *compare* folder of our repository.  The results show that the performance of \texttt{librla} is comparable to that of PyTorch's \texttt{svd\_lowrank} and is faster than the interpolatory decomposition \texttt{id\_decomp}.  The library \texttt{librla} is the only package that provides the user the option of three different factorizations.  The factorization speeds of all three methods are comparable to PyTorch's \texttt{svd\_lowrank}.


|: Comparison of SciPy's \texttt{id\_decomp} with \texttt{librla}. []{label="scipy"}

|Problem    | Scipy           | librla             | Speedup |     
+----------+:---------------:+:-----------------:+------------+
|Hilbert matrix   |                |                 |  |
|2000 x 1000|0.0355s         |0.0024s          |14.6 |
| rank 15 approx|            |                |     | 
+----------+:---------------:+:-----------------:+------------+
|Hilbert matrix   |                |                 |  |
|4000 x 2000 | 0.1557s         |0.0046s          |33.9 |
| rank 15 approx|            |                |     | 
+----------+:---------------:+:-----------------:+------------+
|Decaying spectrum matrix   |                |                 |
|800 x 600 | 0.2933s         |0.0146s          |20 |
| tol = 10^{-2}|            |                |     | 
+----------+:---------------:+:-----------------:+------------+
|Rank 15 matrix   |                |                 |   |  
|800 x 500 | 0.0063s         |0.0013s          |4.8 |
| tol = 10^{-8}|            |                |     | 
+----------+:---------------:+:-----------------:+------------+
|Gaussian mixture model   |                |                 |    |    
|400 x 400 | 0.0699s         | 0.0055s          |12.8 |
| tol = 10^{-2}|            |                |     | 


# Mathematics

The algorithm that serves as the foundation of \texttt{librla} is called \texttt{orth\_sketch}. It creates an orthogonal basis of the range of the operator of interest.  The user-specified tolerance portion of the package was heavily influenced by the methods presented in [@Halko:2011;@Tygert:2008;@Liberty:2007].  The user-specified rank portion of the package was influenced by the \texttt{svd\_lowrank} in PyTorch.  Once the orthogonal basis is created, it is possible to create the low-rank QR, SVD or interpolatory decomposition via standard techniques.  For simplicity of presentation, a pseudocode of \texttt{orth\_sketch} is presented in the following subsection. Pseudocodes for all of the factorizations are provided in the \texttt{PSEUDOCODE.md} file found in the *distrib* folder.

Users call the desired factorization subroutine with a linear operator ${\bf A}$ of size $m\times n$, a desired rank and stopping condition ${rtol}$.  The user has the option to use power iteration and specify the number of iterations $power\_iter$ to accelerate convergence. Users are also provided the option to specify a number of oversampling vectors.  Roughly speaking, given a linear operator ${\bf A}$, desired rank or stopping condition $rtol$, specified $block\_size$ and $power\_iter$, \texttt{orth\_sketch} randomly samples the range of ${\bf A}$ to create an orthogonal basis for the range.  The range is sampled by applying ${\bf A}$ to a matrix ${ \Omega }$ of size $n \times block\_size$ whose entries are uniformly randomly sampled from $[-1,1]$.   The oversampling vectors are included in $block\_size$.  In the tolerance option, the $block\_size$ is adaptively increased to ensure the range is sufficiently sampled and includes the number of user specified extra samples.  In the rank option, the size of $\Omega$ is determined by the desired rank and the number of user specified extra samples. The orthogonal basis is formed by taking a pivoted QR factorization of ${\bf A}\Omega$, where $\Omega$ is optionally refined via power iteration with reorthogonalization at each step.  The magnitudes of the diagonal entries of ${\bf R}$ are stored in a vector ${\bf diagR}$ and are used in the stopping criterion for the tolerance option of the factorization.  \texttt{orth\_sketch} returns ${\bf Q}$ or a submatrix of ${\bf Q}$, ${\bf diagR}$ and an error flag to the factorization routine that called it.  The matrix ${\bf Q}$ is then used to create the desired factorization.  If the sketching step terminates early, the factorization routine falls back to a deterministic algorithm.


## Pseudocode for the \texttt{orth\_sketch} algorithm


```
function orth_sketch(A, rtol, block_size, power_iter):
    Input: A (a real or complex matrix (or linear operator) of size {m×n}),
             rtol (tolerance or rank), block_size, power_iter
    Output: Q (orthonormal basis), flag, diagR

    if rtol ≥ 1:  # Rank mode
        k_max = floor(rtol)
        Ω = random_matrix(n, block_size)        # Uniform[-1,1]
        Ω = power_iteration(A, Ω, power_iter)   # Optional: (A^H A)^p Ω
        Y = A Ω
        Q, R, _ = qr_pivoted(Y)
        return Q[:, 1:k_max], 0, |diag(R)|

    # Tolerance mode (rtol < 1): adaptive rank
    while true:
        Ω = random_matrix(n, block_size)        # Uniform[-1,1]
        Ω = power_iteration(A, Ω, power_iter)   # Optional: (A^H A)^p Ω
        Y = A Ω
        Q, R, _ = qr_pivoted(Y)

        diagR = |diag(R)|
        if diagR[end] / diagR[1] ≤ rtol:
            return Q, 0, diagR

        block_size = min(4 × block_size, min(m, n))
        if block_size ≥ min(m, n):
            return empty(m, 0), 1, []  # Early termination
```

## The different factorization options

There are three different factorization options.  This section provides details on what the different factorizations accomplish.  The demo section that follows explains where to look in the library for demonstrations for calling the available factorizations.  The factorization options are:


\begin{itemize}
\item { \bf QR factorization via  \texttt{qr\_sketch}:} The subroutine returns two matrices: ${\bf Q}$ and ${\bf R}$, and a vector ${\bf p}$.  The rank of the factorization is $k$, where $k\leq \min\{m,n\}$. The $n$ entries of ${\bf p}$ are the list of the column pivots.  The matrix ${\bf Q}$ is of size $m \times k$ and the columns form an orthogonal basis for the range of ${\bf A}$.  The matrix ${\bf R}$ is an upper triangular $k\times n$ matrix. The factorization satisfies

$${\bf{A}}(:,p) \sim {\bf Q \ R}.$$


\item { \bf SVD via \texttt{svd\_sketch:}} The subroutine returns two matrices: ${\bf U}$ and ${\bf V}$ and a vector ${\bf s}$.  The rank of the factorization is $k$, where $k\leq \min\{m,n\}$.  The vector ${\bf s}$ has $k$ entries that are the singular values of ${\bf A}$.  The matrix ${\bf U}$ is of size $m \times k$ and contains the left singular vectors of ${\bf A}$.  The matrix ${\bf V}$ is of size $n\times k$ and contains the right singular vectors of ${\bf A}$.  The columns of both ${\bf U}$ and ${\bf V}$ are orthonormal.  The factorization satisfies

$$ {\bf A}  \sim  {\bf U}\,{\tt diag}({\bf s})\,{\bf V}^{*},$$

where ${\tt diag}({\bf s})$ is a diagonal matrix with non-zero entries coming from the vector ${\bf s}$.

\item { \bf Interpolatory factorization via \texttt{id\_sketch}:} The subroutine returns the number of skeleton columns $k$, a vector ${\bf piv}$ of size $1\times n$ and a matrix ${\bf T}$ of size $k \times (n-k)$.  The first $k$ entries of ${\rm piv}$ denote the skeleton columns; the remaining entries remain in natural order.  The matrix ${\bf T}$ is called the interpolation matrix.  The approximation satisfies the following;


$$ {\bf A}(:, {\bf piv}(k+1:end))\sim {\bf A}(:, {\bf piv}(1:k))\, {\bf T}.$$

To recover an approximation of the full matrix, the interpolation matrix ${\bf W}$ of size $k \times n$ is built by:
 $${\bf W}(:,{\bf piv}(1:k)) = {\bf I}_k,$$
  where ${\bf I}_k$  is an identity matrix of size $k$ and 
$${\bf W}(:,{\bf piv}(k+1:end)) = {\bf T}.$$  
The result is that

$${\bf A} \sim {\bf A}(:, {\bf piv}(1:k))\, {\bf W}.$$


\end{itemize}



# Demo codes

The file named *demo* located in each of the language files provides a collection of codes that demonstrate how to call the factorizations in 'librla' with the different options.  These \texttt{demo\_} codes are designed to aid users.   The codes in the *test* directory named \texttt{test\_} validate that the codes are working correctly.  The code \texttt{test\_all} validates all the factorization techniques while the other test codes are written to test one factorization technique.

# Examples from the literature

To illustrate the ability of the library to handle problems of interest, the techniques are applied to two problems.  The first problem is a data compression problem taken directly from [@Tropp:2019].  The second problem is an image compression problem from [@Duersch:2020].   Examples similar to these are found throughout the randomized linear algebra literature.


Figure \ref{fig:climate_singular} illustrates the performance of the randomized SVD for NOAA ERSST v5 data [@Huang:2017], following an experiment in [@Tropp:2019]. The results were generated by running the \texttt{test\_sst\_mode} code in the *climate_analysis* folder with different variations of the option settings.  The specific options used are (a) none, (b) 10 extra samples, (c) two power iterations, and (d) 10 extra samples and two power iterations.  The extra samples alone do not help the randomized SVD much.  The power iteration is more helpful.  The combination of the two options provides the best approximations of the singular values.


![Illustration of the exact singular values vs approximate singular values via the (a) randomized SVD, (b) 10 extra samples, (c) two power iterations, and (d) 10 extra samples with 2 power iterations.  The example is taken from [@Tropp:2019].  The data is from NOAA Extended Reconstructed Sea Surface Temperature (ERSST), Version 5 [@Huang:2017].  \label{fig:climate_singular}](svdAll.png){width=100%}


Figure \ref{fig:sst_modes} shows the first five modes from the randomized SVD with 10 extra samples and 2 power iterations (empirical orthogonal function (EOF) spatial patterns and principal component (PC) time series) of the NOAA ERSST v5 data.

![Randomized SVD modes of NOAA ERSST v5 sea surface temperature data [@Huang:2017], computed with 10 extra samples and 2 power iterations: spatial empirical orthogonal function (EOF) patterns (left) and principal component (PC) time series (right) for the first five modes.  EOF1 captures the mean spatial pattern, EOF2--4 represent seasonal variations, and EOF5 captures the El Niño–Southern Oscillation (ENSO) signal.  \label{fig:sst_modes}](sstModes.png){width=90%}



The image compression example \texttt{test\_image\_id} code in the *image_analysis* folder produces 6 images.  Figure \ref{fig:image_orig} illustrates five of them: (a) the original image and rank 30 approximations of the image using (b) the randomized SVD, (c) the interpolatory decomposition, (d) the randomized SVD with 15 extra samples and 2 power iterations and (e) the interpolatory decomposition with 15 extra samples and 2 power iterations.  While the rank of the approximations is the same, the SVD produces an image with less smearing.  The examples with oversampling and power iterations further highlight this result.


![Illustration of the use of randomized factorization for image compression: (a) original image, and rank 30 approximations using (b) the randomized SVD, (c) the interpolatory decomposition, (d) the randomized SVD with 15 extra samples and 2 power iterations and (e) the interpolatory decomposition with 15 extra samples and 2 power iterations.  The example is taken from [@Duersch:2020].  The original image is ["Silver Metal Round Gears"](https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/) from Pexels (Creative Commons Zero license).  \label{fig:image_orig}](imageEX.png){width=90%}



# AI usage disclosure

Generative AI tools were used during the preparation of this work to assist with code development, testing, and editing of the manuscript. The authors reviewed and verified all AI-generated content and take full responsibility for the accuracy and integrity of the final publication. The original algorithm was prototyped and debugged in Python. Claude Code (Anthropic, Claude Sonnet 3.5 and Claude Opus 4.6) was used to assist in porting to Matlab and Julia, while maintaining consistency in API calls and documentation. Unit tests were generated to ensure correctness of the ported code.

# Acknowledgements

The work by A. Gillman was supported by the National Science Foundation (DMS-2110886), and a Knut and Alice Wallenberg Foundation Grant. Part of this work was carried out while A. Gillman was in residence at Institut Mittag-Leffler in Djursholm, Sweden in autumn 2025, supported by the Swedish Research Council under grant no. 2021-06594.

Contributions by staff of NIST, an agency of the U.S. Government, are not subject to copyright within the United States.

Certain commercial software and equipment are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by NIST, nor does it imply that the software or equipment identified is necessarily the best available for the purpose.

# References
