---
title: '???: A library of radomized rank revealing factorization algorithms'
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

Rank revealing factorizations have become a vital tool for a variety of algorithms including fast direct solvers and data science.   'package name' provides rank revealing QR, rank revealing SVD and interpolatory decomposition written natively in Python, Julia and Matlab.  The technique is based on the standard randomized rank revealing algorithms but was created to exploit Blas 3 calls. Thus it provides efficient and robust software that has been lacking in these high level languages.


# Statement of need

`package name` is a randomized projection based rank revealing factorization package with Python, Julia and Matlab implementations.  Rank revealing factorizations are useful tools for solving total least squares problems, rank deficient least squares problem, doing matrix approximation, and skeletonizing (i.e. subset selection) a matrix [@RRQRapp].  

One specific application of interest to the authors is of rank revealing factorizations in the development and use of fast direct solvers.  When these solvers are used in a high level language, many would call a Fortran package [@2008_tygert_ID_package] via a specialized wrapper.  The need of the wrapper slows the design and use of these solvers.  `package name` removes this hurdle allowing users to run solver codes natively in the high level language. 


# Mathematics

The algorithm that is implemented is a variant of existing randomized techniques for rank revealing QR factorizations.  The algorithm was heavily influenced by the method presented in [`@2011Halko`].    Add references to other related work.

The idea behind the algorithm is simple.  The range of the matrix $\bf{A}$ can be randomly sampled by applying $\bf{A}$ to a collection of $m$ random vectors arranged as a matrix $\mtx{X}$.   By looking at the diagonal entries of QR factorization of $\bf{A}\bf{X}$, you can determine if $m$ vectors was large enough to capture the full range of $\bf{A}$.  If it is not, the number of random vectors can be increased.  We choose to increase by a factor of 4. This is continued until the range is captured to the desired accuracy.  The result is the approximate rank of $\bf{A}$ denoted $k$ and the columns of $\bf{Q}$ from the QR factorization form a basis for the range of $\bf{A}$.  


Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
