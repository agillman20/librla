---
title: 'Sketcher: A library of radomized rank revealing factorization algorithms'
tags:
  - Python
  - Julia
  - Matlab
  - Rank revealing QR, SVD and Interpolatory decomposition
authors:
  - name: Zydrunas Gimbutas
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Adrianna Gillman
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: National Institute of Standards and Technology (NIST)
   index: 1
 - name: University of Colorado, Boulder, Department of Applied Mathematics
   index: 2
date: 22 September 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Journal of Open Source Software 
#<- The name of the AAS journal.
---

# Summary

Rank revealing factorizations have become a vital tool for a variety of areas including fast direct solvers, reduced order modeling, and data science.   Additionally, rank revealing factorizations are useful tools for solving total least squares problems, rank deficient least squares problem, doing matrix approximation, and skeletonizing (i.e. subset selection) a matrix [@Chan:1992].  'Sketcher' provides rank revealing QR, rank revealing SVD and interpolatory decomposition written natively in Python, Julia and Matlab.  The algorithms ramdomly sampling the range of the matrix or operator a similar manner to '[@Halko:2011]'. A key feature of this package is that it is designed to exploit Level 3 BLAS operators as much as possible.  'Sketcher' is designed for small to mid-range sized matrices (i.e. up to roughly 10,000 in size depending on computing resources).  'Sketcher' is not intendend for matrices that are larger or need to read from hard drive.  

'Sketcher' includes the following options all of which can be used for both real and complex matrices: 

- 'qr_sketch': Rank revealing QR factorization.
- 'svd_sketch': Rank revealing Singular Value Decomposition
- 'id_sketch': Interpolatory Decomposition via randomized sampling.

The package does include the option to create low rank factorizations of matrices that are applied via matrix vector multiplication codes.  'Stetcher' requires both the ability to apply both the matrix and its transpose via a subroutine.  If only application of the matrix is available, 
'Sketcher' will not work.  The following are the options when using a matrix vector multiplication routine:

-
-
-

For each method the user has the option to change the block size for the sampling and to turn on the power option.  The default are block sizes that are multiples of 42 and power option off.

Several test codes are included.  They also serve as demo codes.  The codes include:

While the algorithm used in 'Sketcher' is built in a similar manner to the rank revealing factorizations in [@Halko:2011], there is a large collection of related work [Duersch:2020,Mahoney:2009,MEIER:2024,Martinsson:2019,Sorensen:2016,Gu:1996].  



# Statement of need

While the area of rank revealing factorization techniques, there is not an easy to use widely available factorization library available.  The need is even greater for those using high level languages where standard coding techniques can result in unnecessarily slow code.  The area of fast direct solvers for integral equations is an example of why 'Sketcher' is useful.  For sometime, developers of fast direct solvers using Matlab have been using a Fortran package [@Tygert:2008] based on [@Liberty:2007] via a specialized wrapper.  The need of this wrapper has slowed the design and use of these solvers.  

# Mathematics

We begin with reviewing w

The algorithm that is implemented is a variant of existing randomized techniques for rank revealing QR factorizations.  The algorithm was heavily influenced by the method presented in [`@Halko:2011`].    Add references to other related work.

The idea behind the algorithm is simple.  The range of the matrix $\bf{A}$ can be randomly sampled by applying $\bf{A}$ to a collection of $m$ random vectors arranged as a matrix $\mtx{X}$.   By looking at the diagonal entries of QR factorization of $\bf{A}\bf{X}$, you can determine if $m$ vectors was large enough to capture the full range of $\bf{A}$.  If it is not, the number of random vectors can be increased.  We choose to increase by a factor of 4. This is continued until the range is captured to the desired accuracy.  The result is the approximate rank of $\bf{A}$ denoted $k$ and the columns of $\bf{Q}$ from the QR factorization form a basis for the range of $\bf{A}$.  



# Acknowledgements

The work by A. Gillman was supported by the National Science Foundation (DMS-2110886), and a 
Knut and Alice Wallenberg Foundation Grant. A. Gillman conducted a portion of this work while visiting the Institut Mittag-Leffler.

# References
