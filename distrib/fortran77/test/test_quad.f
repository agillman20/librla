c
c       test_quad: librla77 check for builds promoted to higher precision
c       (make test-quad: -freal-8-real-16 -fdefault-integer-8).
c
c       Small problems on the 100 x 80 Hilbert matrix a(i,j) = 1/(i+j-1)
c       and its complex counterpart, whose singular values fall far below
c       double precision. Tolerances are multiples of the working machine
c       epsilon, so the checks pass only if every operation, constant and
c       intrinsic really runs in the promoted precision. Prints one
c       PASS/FAIL line per check; stops with status 1 on any failure.
c
c       Author: Adrianna Gillman, Zydrunas Gimbutas
c       SPDX-License-Identifier: MIT
c       Assisted by: Claude Code (Anthropic)
c
        program test_quad
        implicit real *8 (a-h,o-z)
        parameter (m=100,n=80,lw=200000)
        dimension a(m,n),u(m,n),vt(n,n),s(n),s0(n),t(n,n),w(lw)
        dimension q(m,n),r(n,n),x(6),ref(6)
        dimension iseed(4),ipiv(n)
        complex *16 ac(m,n),uc(m,n),vtc(n,n),wc(lw)
        common /tstcnt/ npass,nfail
        save
        data ref/3.7327920546847082d-01,8.2093410748050388d-01,
     1      5.5866811353917711d-01,6.4291221902741569d-01,
     2      6.8760847451716955d-01,1.6449965895444763d-01/
c
        npass=0
        nfail=0
        eps=rla_eps()
        call check('working precision is promoted (eps < 1e-30)',
     1      eps .lt. 1.0d-30)
c
c       the generator: dlarnv's values are multiples of 2**-48, exact
c       in any precision
c
        call seed4(iseed,1,2,3,5)
        call rla_drandn(iseed,6,x)
        e=0
        ex=0
        do 1100 i=1,6
        e=max(e,abs(x(i)-ref(i)))
        d=(x(i)+1)*2.0d0**47
        ex=max(ex,abs(d-anint(d)))
 1100 continue
        call checkle('rng: dlarnv reference values',e,1.0d-16)
        call check('rng: values are exact multiples of 2**-48',
     1      ex .le. 0 .and. iseed(1) .eq. 2384 .and. iseed(4) .eq. 1229)
c
        do 1300 j=1,n
        do 1200 i=1,m
        a(i,j)=1.0d0/(i+j-1)
        ac(i,j)=dcmplx(1.0d0,(i-j)/80.0d0)/(i+j-1)
 1200 continue
 1300 continue
        anrm=fro(m,n,a)
c
c       ---------------- full SVD ----------------
c
        call rla_dsvd(m,n,a,m,0.0d0,0.0d0,n,0,k,s0,u,m,vt,n,w,lw,ier)
        call check('rla_dsvd full: ier 0, k = 80',ier .eq. 0 .and.
     1      k .eq. n)
        call checkle('rla_dsvd full: ||A - U S Vt|| / (eps ||A||)',
     1      svderr(m,n,a,k,u,s0,vt)/(eps*anrm),100.0d0)
        call checkle('rla_dsvd full: orthonormality / eps',
     1      max(ortc(m,k,u),ortr(k,n,vt))/eps,1.0d3)
        call check('rla_dsvd full: s_80 / s_1 below 1e-30 (needs quad)',
     1      s0(n) .lt. 1.0d-30*s0(1))
c
c       ---------------- truncated SVD, early exit ----------------
c
        rtol=1.0d-25
        call rla_dsvd(m,n,a,m,rtol,0.0d0,n,2,k,s,u,m,vt,n,w,lw,ier)
        kexp=0
        do 2100 i=1,n
        if( s0(i) .ge. rtol*s0(1) ) kexp=i
 2100 continue
        e=0
        do 2200 i=1,k
        e=max(e,abs(s(i)-s0(i)))
 2200 continue
        call check('rla_dsvd rtol 1e-25, iexit 2: rank as the full SVD',
     1      ier .eq. 0 .and. k .eq. kexp .and. k .lt. n)
        call checkle('rla_dsvd rtol 1e-25: |s - s_full| / (rtol s1)',
     1      e/(rtol*s0(1)),0.02d0)
        call checkle('rla_dsvd rtol 1e-25: error / s(k+1)',
     1      svderr(m,n,a,k,u,s,vt)*anrm/s0(k+1),2.0d0)
c
c       ---------------- sketches at rtol 1e-25 ----------------
c
        call seed4(iseed,1,2,3,5)
        call rla_dsvd_sketch(m,n,a,m,rtol,42,0,12,iseed,n,k,u,m,s,vt,
     1      n,w,lw,ier)
        call checkle('svd_sketch rtol 1e-25: error / rtol',
     1      svderr(m,n,a,k,u,s,vt)/rtol,10.0d0)
        call seed4(iseed,1,2,3,5)
        call rla_dqr_sketch(m,n,a,m,rtol,42,0,12,iseed,n,k,q,m,r,n,
     1      ipiv,w,lw,ier)
        call checkle('qr_sketch rtol 1e-25: error / rtol',
     1      qrerr(m,n,a,k,q,r,ipiv)/rtol,10.0d0)
        call seed4(iseed,1,2,3,5)
        call rla_did_sketch(m,n,a,m,rtol,42,0,12,1,iseed,n,k,ipiv,t,n,
     1      w,lw,ier)
        e=fiderr(m,n,a,k,ipiv,t)/rtol
        if( k .ge. n ) e=1.0d30
        call checkle('id_sketch rtol 1e-25, k < 80: error / rtol',e,
     1      1.0d3)
        call rla_did_qrpiv(m,n,a,m,rtol,1,n,k,ipiv,t,n,w,lw,ier)
        e=fiderr(m,n,a,k,ipiv,t)/rtol
        if( k .ge. n ) e=1.0d30
        call checkle('id_qrpiv  rtol 1e-25, k < 80: error / rtol',e,
     1      1.0d3)
c
c       ---------------- pivoted QR, early exit ----------------
c
        call tcopy(m,n,a,q)
        call rla_dqrcp(m,n,n,-1.0d0,1.0d-25,q,m,k,rk,rrel,ipiv,w,
     1      w(n+1),ier)
        call check('rla_dqrcp reltol 1e-25: stops early',
     1      ier .eq. 0 .and. k .lt. n .and. rrel .le. 1.0d-25)
c
c       ---------------- complex ----------------
c
        acn=0
        do 3200 j=1,n
        do 3100 i=1,m
        acn=acn+abs(ac(i,j))**2
 3100 continue
 3200 continue
        acn=sqrt(acn)
        call rla_zsvd(m,n,ac,m,0.0d0,0.0d0,n,0,k,s,uc,m,vtc,n,wc,lw,
     1      ier)
        call checkle('rla_zsvd full: ||A - U S Vh|| / (eps ||A||)',
     1      zsverr(m,n,ac,k,uc,s,vtc)/(eps*acn),100.0d0)
        call seed4(iseed,1,2,3,5)
        call rla_zsvd_sketch(m,n,ac,m,rtol,42,1,12,iseed,n,k,uc,m,s,
     1      vtc,n,wc,lw,ier)
        call checkle('zsvd_sketch rtol 1e-25: error / rtol',
     1      zsverr(m,n,ac,k,uc,s,vtc)/rtol,10.0d0)
c
        write(*,'(a,i3,a,i3,a)') 'SUMMARY ',npass,' passed, ',nfail,
     1      ' failed'
        if( nfail .gt. 0 ) stop 1
        end
c
c
c
c
c
        subroutine check(name,ok)
        character *(*) name
        logical ok
        common /tstcnt/ npass,nfail
c
        if( ok ) then
        npass=npass+1
        write(*,'(2a)') 'PASS ',name
        else
        nfail=nfail+1
        write(*,'(2a)') 'FAIL ',name
        endif
        return
        end
c
c
c
c
c
        subroutine checkle(name,val,tol)
        implicit real *8 (a-h,o-z)
        character *(*) name
        character *40 buf
        character *160 msg
c
        write(buf,'(a,e9.2,a,e8.1,a)') ' (',val,' <= ',tol,')'
        msg=name//buf
        l=len(name)+40
 1100   if( l .gt. 1 .and. msg(l:l) .eq. ' ' ) then
        l=l-1
        goto 1100
        endif
        call check(msg(1:l),val .le. tol)
        return
        end
c
c
c
c
c
        subroutine seed4(iseed,i1,i2,i3,i4)
        dimension iseed(4)
c
        iseed(1)=i1
        iseed(2)=i2
        iseed(3)=i3
        iseed(4)=i4
        return
        end
c
c
c
c
c
        subroutine tcopy(m,n,a,b)
        implicit real *8 (a-h,o-z)
        dimension a(m,n),b(m,n)
c
        do 1200 j=1,n
        do 1100 i=1,m
        b(i,j)=a(i,j)
 1100 continue
 1200 continue
        return
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
        real *8 function svderr(m,n,a,k,u,s,vt)
        implicit real *8 (a-h,o-z)
        dimension a(m,n),u(m,*),s(*),vt(n,*)
c
c       ||A - U(:,1:k) diag(s) Vt(1:k,:)||_F / ||A||_F (vt has ld n)
c
        e=0
        do 1400 j=1,n
        do 1300 i=1,m
        d=a(i,j)
        do 1200 l=1,k
        d=d-u(i,l)*s(l)*vt(l,j)
 1200 continue
        e=e+d**2
 1300 continue
 1400 continue
        svderr=sqrt(e)/fro(m,n,a)
        return
        end
c
c
c
c
c
        real *8 function zsverr(m,n,a,k,u,s,vt)
        implicit real *8 (a-h,o-z)
        complex *16 a(m,n),u(m,*),vt(n,*),d
        dimension s(*)
c
c       complex version of svderr
c
        e=0
        f=0
        do 1400 j=1,n
        do 1300 i=1,m
        d=a(i,j)
        f=f+abs(d)**2
        do 1200 l=1,k
        d=d-u(i,l)*s(l)*vt(l,j)
 1200 continue
        e=e+abs(d)**2
 1300 continue
 1400 continue
        zsverr=sqrt(e/f)
        return
        end
c
c
c
c
c
        real *8 function qrerr(m,n,a,k,q,r,ipiv)
        implicit real *8 (a-h,o-z)
        dimension a(m,n),q(m,*),r(n,*),ipiv(*)
c
c       ||A(:,ipiv) - Q(:,1:k) R(1:k,:)||_F / ||A||_F (r has ld n)
c
        e=0
        do 1400 j=1,n
        do 1300 i=1,m
        d=a(i,ipiv(j))
        do 1200 l=1,k
        d=d-q(i,l)*r(l,j)
 1200 continue
        e=e+d**2
 1300 continue
 1400 continue
        qrerr=sqrt(e)/fro(m,n,a)
        return
        end
c
c
c
c
c
        real *8 function fiderr(m,n,a,k,ipiv,t)
        implicit real *8 (a-h,o-z)
        dimension a(m,n),t(n,*),ipiv(*)
c
c       ||A(:,ipiv(k+1:n)) - A(:,ipiv(1:k)) T||_F / ||A||_F (t has ld n)
c
        e=0
        do 1400 j=1,n-k
        do 1300 i=1,m
        d=a(i,ipiv(k+j))
        do 1200 l=1,k
        d=d-a(i,ipiv(l))*t(l,j)
 1200 continue
        e=e+d**2
 1300 continue
 1400 continue
        fiderr=sqrt(e)/fro(m,n,a)
        return
        end
c
c
c
c
c
        real *8 function ortc(m,k,q)
        implicit real *8 (a-h,o-z)
        dimension q(m,*)
c
c       max |Q^T Q - I| over the first k columns
c
        ortc=0
        do 1300 j=1,k
        do 1200 i=1,k
        d=0
        do 1100 l=1,m
        d=d+q(l,i)*q(l,j)
 1100 continue
        if( i .eq. j ) d=d-1
        ortc=max(ortc,abs(d))
 1200 continue
 1300 continue
        return
        end
c
c
c
c
c
        real *8 function ortr(k,n,v)
        implicit real *8 (a-h,o-z)
        dimension v(n,*)
c
c       max |V V^T - I| over the first k rows of v (ld n)
c
        ortr=0
        do 1300 j=1,k
        do 1200 i=1,k
        d=0
        do 1100 l=1,n
        d=d+v(i,l)*v(j,l)
 1100 continue
        if( i .eq. j ) d=d-1
        ortr=max(ortr,abs(d))
 1200 continue
 1300 continue
        return
        end
