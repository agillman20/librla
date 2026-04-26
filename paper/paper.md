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
 
date: April 23, 2026
bibliography: paper.bib

# Optional fields if submitting to an AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Journal of Open Source Software 
#<- The name of the AAS journal.
---

# Summary

Randomized linear algebra algorithms have become a vital tool for a variety of areas including fast direct solvers, reduced order modeling, and data science.   Additionally, randomized linear algebra provides useful routines for solving total least squares problems and rank-deficient least squares problems, doing matrix approximation, and skeletonizing (i.e. subset selection) a matrix [@Chan:1992].    \texttt{librla} is designed for small to mid-range sized matrices (i.e. up to roughly 10,000 in size depending on computing resources).  \texttt{librla} is not intended for matrices that are larger or do not fit in memory.

\texttt{librla} includes the following options, all of which can be used for both real and complex matrices: 

- \texttt{qr\_sketch}: Randomized QR factorization.
- \texttt{svd\_sketch}: Randomized Singular Value Decomposition (SVD).
- \texttt{id\_sketch}: Interpolatory Decomposition via randomized sampling.

The user has the choice of specifying a desired accuracy (tolerance) or rank.  For each method, the user has the option to use extra samples and to use a specified number of power iterations.  
The package does include the option to create low-rank factorizations of matrices that are applied via matrix-vector multiplication codes. When the matrix-vector multiplication is "matrix-free," the creation of the low factorization is also matrix free. In order to use this option, \texttt{librla} requires the ability to apply both the matrix and its transpose via a subroutine.



# Statement of need

While there is a large amount of research activity in the field of randomized linear algebra, an easy-to-use, efficient and stable factorization library is not available.  For several factorization libraries, the stability issue manifest itself by either failure in the form of an invalid factorization or being unable to return anything.  When new techniques implemented in a high level language are dependent on low level language libraries, it has a big impact on the portability and usability of the new work.  For example, this problem has impacted of the field of fast direct solvers for boundary integral equations which often depend on a legacy Fortran interpolatory decomposition library [@Tygert:2008].

Currently, there are two related routines in widely available, maintained Python packages.  They are PyTorch's \texttt{svd\_lowrank} and SciPy's \texttt{id\_decomp} found in the \texttt{scipy.linalg.interpolative} library.  Codes allowing the user to compare the performance are available in the *compare* folder of our repository.  To illustrate the performance of librla, a subset of the provided examples are presented here.  The experiments were run on a MacBook Pro with an M4 processor and 64 GB of RAM in the virtual environment created by \texttt{setup\_venv.sh}.    The experiments consider the following matrices: Hilbert matrices, a matrix where the spectrum decays $\sim1/k$ for integer $k$ between 1 and the size of the matrix, and a matrix from a Gaussian mixture model [@Dong:2025]. Table 1 reports a subset of results obtained by running \texttt{compare\_svd\_torch.py}.  For a sequence of matrices, rank 15 approximate singular value decompositions are computed.  This comparison is only for the fixed rank factorizations as that is the only option available in the Pytorch \texttt{svd\_lowrank} software.  Additionally, the singular value decomposition is the only factorization available in Pytorch.  The results demonstrate that the timings are comparable.  Table 2 reports a subset of the result obtained by running \texttt{compare\_id\_scipy.py}.  The Interpolatory decomposition is the only factorization available in SciPy and it allows the user to input a desire rank or tolerance.  The results in the table include both types of experiments.  Table 2 also reports the ratio of time it takes librla to compute the factorization over the time it takes SciPy under the title *Speedup*.   The results illustrate the benefits of using the \texttt{librla} package.  




| Problem                  | Size        | Pytorch   | librla   | Relative Error |
|--------------------------|-------------|-----------|----------|----------------|
| Hilbert matrix           | 2000 x 1000 | 0.002s    | 0.0018s  | 3.679e-08      |
| Hilbert matrix           | 4000 x 2000 | 0.0045s   | 0.0043s  | 1.604e-07      |
| Decaying spectrum matrix | 800 x 600   |  0.0026s  | 0.0023s  | 1.529e-01      |
| Gaussian mixture model   | 400 x 400   | 0.0016s   | 0.0014s  | 7.096e-01      |
: The table reports the relative error and the time in seconds for creating rank 15 factorizations of different matrices using Pytorch and librla.






| Problem                  | Size        | Stopping Condition | SciPy   | librla  | Speedup | Relative Error |
|--------------------------|-------------|--------------------|---------|---------|---------|----------------|
| Hilbert matrix           | 2000 x 1000 | rank 15            | 0.0355s | 0.0024s | 14.6    | 8.571e-08      |
| Hilbert matrix           | 4000 x 2000 | rank 15            | 0.1557s | 0.0046s | 33.9    | 1.306e-06      |
| Decaying spectrum matrix | 800 x 600   | tol  = 0.01        | 0.2933s | 0.0146s | 20      | 0              |
| Gaussian mixture model   | 400 x 400   | tol = 0.01         | 0.0699s | 0.0055s | 12.8    | 5.659e-05      |
: The table reports the relative error and the time in seconds for creating low factorizations of different matrices using SciPy and librla. The speedup over the SciPy package is also reported. The examples involve fixed rank or tolerance approximations. 


# State of the field 

While the algorithm used in \texttt{librla} is built in a similar manner to the randomized methods in [@Halko:2011;@Liberty:2007]. There is a large collection of related work.  [@Martinsson:2020;@Mahoney:2011;@Drineas2017LecturesOR;@Woodruff:2014;@Voronin:2017] are some review papers on the field.  Some literature [@Mahoney:2009;@Sorensen:2016; @Drineas:2008] has focused on the design of CUR factorizations.  While other manuscripts [@Frieze:2004;@Drineas:2006_2;@Drinea:2006_3] have focused on Monte Carlo approaches to creating the low rank factorizations.  Randomized projection methods [@Halko:2011;@Liberty:2007;@Duersch:2020;@MEIER:2024; @Gu:1996; @ERICHSON:2018;@Rokhlin:2008] are common.  Deterministic factorization techniques exist [@Duersch:2017_2;@Feng:2018].  Many techniques are specifically designed for distributed memory  and/or to not require many passes through the data [@Martinsson:2019;@Sarlos:2006;@Tropp:2017;@Yu:2017;@Lindquist:2020;@Yang:2015;@Bjarkason:2019].  There is much literature from the randomized linear algebra community for applications including $L_2$ solutions [@Drineas:2006;@Drineas:2007;@Avron:2010; @Meng:2014;@Pilanci:2016], reduced order modeling [@Sorensen:2016], linear programming [@Drineas:2012;@Chowdhury:2020], statistics [@Drineas:2016], machine learning [@Yao:2021],
Newton methods [@Roosta-Khorasani:2019; @Pilanci:2017], and differentially private matrices [@Balu:2016; @Cormode:2018; @Choi:2020].
The user-specified rank algorithm in \texttt{librla} is influenced by the \texttt{svd\_lowrank} routine in PyTorch.


# Mathematics

The algorithm that serves as the foundation of \texttt{librla} is called \texttt{orth\_sketch}. It creates an orthogonal basis of the range of the operator of interest.  The user-specified tolerance portion of the package was heavily influenced by the methods presented in [@Halko:2011;@Tygert:2008;@Liberty:2007].  The user-specified rank portion of the package was influenced by the \texttt{svd\_lowrank} in PyTorch.  Once the orthogonal basis is created, it is possible to create the low-rank QR, SVD or interpolatory decomposition via standard techniques. Pseudocodes for all of the factorizations are provided in the \texttt{PSEUDOCODE.md} file found in the *distrib* folder.

Users call the desired factorization subroutine with a linear operator ${\bf A}$ of size $m\times n$, a desired rank and stopping condition ${tol\_or\_rank}$.  The user has the option to use power iteration and specify the number of iterations $power\_iter$ to accelerate convergence. Users are also provided the option to specify a number of oversampling vectors.  Roughly speaking, given a linear operator ${\bf A}$, tolerance condition or desired rank $tol\_or\_rank$, specified $block\_size$ and $power\_iter$, \texttt{orth\_sketch} randomly samples the range of ${\bf A}$ to create an orthogonal basis for the range.  The range is sampled by applying ${\bf A}$ to a matrix ${ \Omega }$ of size $n \times block\_size$ whose entries are uniformly randomly sampled from $[-1,1]$.   This choice of matrix entries for $\Omega$ ensures that the vectors are linearly independent but may not be the optimal basis choice.  (This is an open question.)  The size $\Omega$ is set so that number of columns includes the over sampling vectors; i.e. the parameter $block\_size$ is updated to be the sum of the original $block\_size$ and the number of oversampling vectors.  In the tolerance option, the $block\_size$ is adaptively increased until the stopping criterion is satisfied.    In the rank option, the size of $\Omega$ is determined by the desired rank and the number of user specified extra samples. The orthogonal basis is formed by taking a pivoted QR factorization of ${\bf A}\Omega$, where $\Omega$ is optionally refined via power iteration with reorthogonalization at each step.  The magnitudes of the diagonal entries of ${\bf R}$ are stored in a vector ${\bf diagR}$ and are used in the stopping criterion for the tolerance option of the factorization.  \texttt{orth\_sketch} returns ${\bf Q}$ or a submatrix of ${\bf Q}$, ${\bf diagR}$ and an error flag to the factorization routine that called it.  The matrix ${\bf Q}$ is then used to create the desired factorization.  In the tolerance option of \texttt{librla}, the range of ${\bf A}$ is considered sufficiently if the ratio of the magnitude of the largest and smallest diagonal entries of the ${\bf R}$ is less than the specified tolerance ${tol\_or\_rank}$.  If the sketching step terminates early, the factorization routine falls back to a deterministic algorithm.


# Software design

The library \texttt{librla}, written natively in Python, Julia and Matlab, provides low-rank QR factorizations, SVDs and interpolatory decompositions.  The algorithms randomly sample the range of the matrix or operator in a similar manner to [@Halko:2011]. A key feature of this package is that it is designed to exploit Level 3 BLAS operators as much as possible.  The use of Level 3 BLAS allows for all the linear algebraic operations in the method be executed via low level optimized code.  This design choice is the foundation for the efficiency of the software.  In fact, the performance in terms of both speed and stability is the same in all three languages.  The codes are consistent between languages allowing for the software to be easily maintained.

# Research impact statement

The avialability of efficient and stable randomized factorization techniques has the potential to impact various research areas.  For example, the \texttt{librla} library allows for the fast direct solvers the authors develop to be easily shared with collaborators.  Additionally, the \texttt{librla} library allows for easy reproducibility of the solver work as it is no longer dependent on platform dependent wrappers on FORTRAN code.  Other areas that can experience similar benefits include reduced order modeling and data compression.


# AI usage disclosure

Generative AI tools were used during the preparation of this work to assist with code development, testing, and editing of the manuscript. The authors reviewed and verified all AI-generated content and take full responsibility for the accuracy and integrity of the final publication. The original algorithm was prototyped and debugged in Python. Claude Code (Anthropic, Claude Sonnet 3.5 and Claude Opus 4.6) was used to assist in porting to Matlab and Julia, while maintaining consistency in API calls and documentation. Unit tests were generated to ensure correctness of the ported code.

# Acknowledgements

The work by A. Gillman was supported by the National Science Foundation (DMS-2110886), and a Knut and Alice Wallenberg Foundation Grant. Part of this work was carried out while A. Gillman was in residence at Institut Mittag-Leffler in Djursholm, Sweden in autumn 2025, supported by the Swedish Research Council under grant no. 2021-06594.

Contributions by staff of NIST, an agency of the U.S. Government, are not subject to copyright within the United States.

Certain commercial software and equipment are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by NIST, nor does it imply that the software or equipment identified is necessarily the best available for the purpose.

# References
