c
c       demo_basic: librla77 on a dense Hilbert-like matrix and a
c       matrix-free operator
c
c       Author: Adrianna Gillman, Zydrunas Gimbutas
c       SPDX-License-Identifier: MIT
c       Assisted by: Claude Code (Anthropic)
c
        program demo_basic
        implicit real *8 (a-h,o-z)
        parameter (m=1000,n=500,kc=500,lw=2000000)
        parameter (mb=2000,nb=1500,ir=12)
        dimension a(m,n),u(m,kc),s(kc),vt(kc,n),q(m,kc),r(kc,n),
     1      t(kc,n),w(lw),ipiv(n),iseed(4),ipar(1)
        dimension bc(mb*ir+ir*nb),ub(mb,ir),vtb(ir,nb)
        complex *16 zdum(1)
        character *16 version
        external product
        save
c
c       dense: a(i,j) = 1/(i+j-1), 1000 x 500
c
        do 1200 j=1,n
        do 1100 i=1,m
        a(i,j)=1.0d0/(i+j-1)
 1100 continue
 1200 continue
        call rla77_version(version)
        write(*,'(2a)') 'librla Fortran 77 port ',
     1      version(1:len_trim(version))
c
c       the workspace query (lw = -1) returns the size needed in w(1)
c
        iseed(1)=0
        iseed(2)=0
        iseed(3)=0
        iseed(4)=1
        call rla_dsvd_sketch(m,n,a,m,1.0d-10,42,0,12,iseed,kc,k,u,m,s,
     1      vt,kc,w,-1,ier)
        write(*,'(a,i0,a)') 'svd_sketch  rtol=1e-10: workspace ',
     1      nint(w(1)),' doubles'
        call rla_dsvd_sketch(m,n,a,m,1.0d-10,42,0,12,iseed,kc,k,u,m,s,
     1      vt,kc,w,lw,ier)
        err=0
        do 1500 j=1,n
        do 1400 i=1,m
        d=a(i,j)
        do 1300 l=1,k
        d=d-u(i,l)*s(l)*vt(l,j)
 1300 continue
        err=err+d**2
 1400 continue
 1500 continue
        write(*,'(a,i0,a,1p,e9.2)') 'svd_sketch  rtol=1e-10: k = ',k,
     1      ', ||A - U S Vt||_F / ||A||_F = ',sqrt(err)/fro(m,n,a)
c
        call rla_dqr_sketch(m,n,a,m,20.0d0,42,1,12,iseed,kc,k,q,m,r,kc,
     1      ipiv,w,lw,ier)
        err=0
        do 1800 j=1,n
        do 1700 i=1,m
        d=a(i,ipiv(j))
        do 1600 l=1,min(j,k)
        d=d-q(i,l)*r(l,j)
 1600 continue
        err=err+d**2
 1700 continue
 1800 continue
        write(*,'(a,i0,a,1p,e9.2)') 'qr_sketch   rank 20:    k = ',k,
     1      ', ||A(:,p) - Q R||_F / ||A||_F = ',sqrt(err)/fro(m,n,a)
c
c       method 3 = lstsq; it materializes A, so it needs more workspace
c
        call rla_did_sketch(m,n,a,m,1.0d-10,42,0,12,3,iseed,kc,k,ipiv,
     1      t,kc,w,-1,ier)
        write(*,'(a,i0,a)') 'id_sketch   rtol=1e-10: workspace ',
     1      nint(w(1)),' doubles'
        call rla_did_sketch(m,n,a,m,1.0d-10,42,0,12,3,iseed,kc,k,ipiv,
     1      t,kc,w,lw,ier)
        if( ier .ne. 0 ) write(*,'(a,i0)') 'id_sketch: ier = ',ier
        err=0
        do 2100 j=1,n-k
        do 2000 i=1,m
        d=a(i,ipiv(k+j))
        do 1900 l=1,k
        d=d-a(i,ipiv(l))*t(l,j)
 1900 continue
        err=err+d**2
 2000 continue
 2100 continue
        write(*,'(a,i0,a,1p,e9.2)') 'id_sketch   rtol=1e-10: k = ',k,
     1      ', ID error = ',sqrt(err)/fro(m,n,a)
c
c       matrix-free: rank-12 product of 2000 x 12 and 12 x 1500 factors
c       stored back to back in rpar = bc, applied by subroutine product
c
        iseed(1)=0
        iseed(2)=0
        iseed(3)=0
        iseed(4)=2027
        call rla_drandn(iseed,mb*ir+ir*nb,bc)
        ipar(1)=ir
        zdum(1)=0
        call rla_dsvd_sketchf(mb,nb,product,ipar,bc,zdum,12.0d0,42,0,
     1      12,iseed,ir,k,ub,mb,s,vtb,ir,w,lw,ier)
        err=0
        ref=0
        do 2400 j=1,nb
        do 2300 i=1,mb
        d=0
        do 2200 l=1,ir
        d=d+bc(i+(l-1)*mb)*bc(mb*ir+l+(j-1)*ir)-ub(i,l)*s(l)*vtb(l,j)
 2200 continue
        err=err+d**2
 2300 continue
 2400 continue
        do 2600 j=1,nb
        do 2500 i=1,mb
        d=0
        do 2450 l=1,ir
        d=d+bc(i+(l-1)*mb)*bc(mb*ir+l+(j-1)*ir)
 2450 continue
        ref=ref+d**2
 2500 continue
 2600 continue
        write(*,'(a,i0,a,1p,e9.2)')
     1      'matrix-free svd_sketch rank 12: k = ',k,', error = ',
     2      sqrt(err/ref)
        write(*,'(a,1p,3e12.4)') '  leading singular values:',
     1      (s(i),i=1,3)
        end
c
c
c
c
c
        real *8 function fro(m,n,a)
        implicit real *8 (a-h,o-z)
        dimension a(m,n)
c
        fro=0
        do 1200 j=1,n
        do 1100 i=1,m
        fro=fro+a(i,j)**2
 1100 continue
 1200 continue
        fro=sqrt(fro)
        return
        end
c
c
c
c
c
        subroutine product(itrans,m,n,k,x,y,ipar,rpar,zpar)
        implicit real *8 (a-h,o-z)
        dimension x(*),y(*),ipar(*),rpar(*)
        complex *16 zpar(*)
c
c       A = B C with B (m x r) then C (r x n) stored in rpar, r = ipar(1):
c       itrans=0: y(m,k) = B (C x); itrans=1: y(n,k) = C^T (B^T x)
c
        ir=ipar(1)
        if( itrans .eq. 0 ) then
        call prod0(m,n,ir,k,rpar,rpar(m*ir+1),x,y)
        else
        call prod1(m,n,ir,k,rpar,rpar(m*ir+1),x,y)
        endif
        if( abs(zpar(1)) .lt. 0 ) y(1)=0
        return
        end
c
c
c
c
c
        subroutine prod0(m,n,ir,k,b,c,x,y)
        implicit real *8 (a-h,o-z)
        dimension b(m,ir),c(ir,n),x(n,k),y(m,k),tmp(64)
c
        do 1500 j=1,k
        do 1200 l=1,ir
        tmp(l)=0
        do 1100 i=1,n
        tmp(l)=tmp(l)+c(l,i)*x(i,j)
 1100 continue
 1200 continue
        do 1400 i=1,m
        y(i,j)=0
        do 1300 l=1,ir
        y(i,j)=y(i,j)+b(i,l)*tmp(l)
 1300 continue
 1400 continue
 1500 continue
        return
        end
c
c
c
c
c
        subroutine prod1(m,n,ir,k,b,c,x,y)
        implicit real *8 (a-h,o-z)
        dimension b(m,ir),c(ir,n),x(m,k),y(n,k),tmp(64)
c
        do 1500 j=1,k
        do 1200 l=1,ir
        tmp(l)=0
        do 1100 i=1,m
        tmp(l)=tmp(l)+b(i,l)*x(i,j)
 1100 continue
 1200 continue
        do 1400 i=1,n
        y(i,j)=0
        do 1300 l=1,ir
        y(i,j)=y(i,j)+c(l,i)*tmp(l)
 1300 continue
 1400 continue
 1500 continue
        return
        end
