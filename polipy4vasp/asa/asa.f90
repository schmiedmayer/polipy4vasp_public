
!*********************************************************************
! RCS:  $Id: asa.F,v 1.7 2003/06/27 13:22:13 kresse Exp kresse $
!
!  this modul contains code to calculate Clebsch-Gordan coefficients
!  and code to evaluate the integrals of three spherical (real or
!  complex) harmonics
!
!  The part which calculates the Clebsch-Gordan coefficients
!  was kindly supplied by Helmut Nowotny
!  and is taken from an ASW (augmented spherical waves) code
!  written by Williams, Kuebler and Gelat.
!  port to f90, a lot of extensions and code clean up was
!  done by gK (I think I have also written the version for the real
!  spherical harmonics, but I am not shure ;-).
!
!*********************************************************************


MODULE asa
    
!    USE iso_c_binding, ONLY: c_double, c_int

! public variables
      INTEGER,PUBLIC, PARAMETER :: NCG=13000 ! LMAX= 6 (f electrons)
!     (LMAX=8: NCG=45000, LMAX=10: NCG=123000, LMAX=14: NCG=588000, LMAX=20: NCG=3191000)
    
      REAL(8),PUBLIC :: FAC(40)    ! table for factorials
      REAL(8),PUBLIC :: YLM3(NCG)  ! table which contains the intregral of three
                          ! real spherical harmonics
      REAL(8),PUBLIC :: YLM3I(NCG) ! inverse table
      INTEGER,PUBLIC :: JL(NCG)    ! index L for each element in the array YLM3
      INTEGER,PUBLIC :: JS(NCG)    ! compound index L,M for each element in the array YLM3
                                   ! JS =  L*(L+1)+M+1 (where M=-L,...,L)
      INTEGER,PUBLIC :: INDCG(NCG) ! index into array YLM3 which gives the starting
                                   ! position for one l,m ,lp,mp  quadruplet
      INTEGER :: LMAXCG=-1         ! maximum l
      INTEGER,SAVE,PUBLIC :: YLM3LOOKUP_TABLE(0:6,0:6) ! LMAX = 6

  CONTAINS

!**************** FUNCTION CLEBGO ************************************
!
! caculate Clebsch-Gordan-coeff. <J1 J2 M1 M2 I J3 M3>
! using racah-formel
! FAC is a user supplied array containing factorials
!
!*********************************************************************

  FUNCTION CLEBGO(J1,J2,J3,M1,M2,M3)

      IMPLICIT NONE
      INTEGER J1,J2,J3,M1,M2,M3
      REAL(8) CLEBGO

!f2py intent(in)  J1,J2,J3,M1,M2,M3
!f2py intent(out) CLEBGO

! local variables
      REAL(8) T,T1
      INTEGER K1,K2,K3,K4,K5,K6,M,N,N1,N2

      IF(M3/=M1+M2) GO TO 2
      K1=J1+J2-J3+1
      K2=J3+J1-J2+1
      K3=J3+J2-J1+1
      K4=J1+J2+J3+2
      T= (2*J3+1)*FAC(K1)*FAC(K2)*FAC(K3)/FAC(K4)
      K1=J1+M1+1
      K2=J1-M1+1
      K3=J2+M2+1
      K4=J2-M2+1
      K5=J3+M3+1
      K6=J3-M3+1
      T=SQRT(T*FAC(K1)*FAC(K2)*FAC(K3)*FAC(K4)*FAC(K5)*FAC(K6))
      N1=MAX0(J2-J3-M1,J1-J3+M2,0)+1
      N2=MIN0(J1+J2-J3,J1-M1,J2+M2)+1
      IF(N1>N2) GO TO 2
      T1=0.0_8
      DO M=N1,N2
         N=M-1
         K1=J1+J2-J3-N+1
         K2=J1-M1-N+1
         K3=J2+M2-N+1
         K4=J3-J2+M1+N+1
         K5=J3-J1-M2+N+1
         T1=T1+ (1+4*(N/2)-2*N)/(FAC(M)*FAC(K1)*FAC(K2)*FAC(K3) &
              &  *FAC(K4)*FAC(K5))
      ENDDO
      CLEBGO=T*T1
      RETURN
! coefficient is zero, drop back
 2    CONTINUE
      CLEBGO=0.0_8
      RETURN

  END FUNCTION CLEBGO

!**************** FUNCTION CLEBG0 ************************************
!
! calculate Clebsch-Gordan-coeff. <L1 L2 0 0 I L3 0>
! using racah-formel
! FAC is a user supplied array containing factorials
!
!*********************************************************************

  FUNCTION CLEBG0(L1,L2,L3)

      IMPLICIT NONE
      
      INTEGER L1,L2,L3
      REAL(8) CLEBG0

!f2py intent(in)  L1,L2,L3
!f2py intent(out) CLEBG0

! local variables
      INTEGER X,P,LT

      LT=L1+L2+L3
      P=LT/2
      IF(2*P/=LT) GO TO 1
      CLEBG0= SQRT( REAL(2*L3+1,KIND=8)/(LT+1))
      CLEBG0=CLEBG0*FAC(P+1)/SQRT(FAC(2*P+1))
      X=P-L1
      CLEBG0=CLEBG0*SQRT(FAC(2*X+1))/FAC(X+1)
      X=P-L2
      CLEBG0=CLEBG0*SQRT(FAC(2*X+1))/FAC(X+1)
      X=P-L3
      CLEBG0=CLEBG0*SQRT(FAC(2*X+1))/FAC(X+1)
      IF(X>2*(X/2)) CLEBG0=-CLEBG0
      RETURN
! coefficient is zero, drop back
 1    CONTINUE
      CLEBG0=0.0_8
      RETURN
  END FUNCTION CLEBG0

!************************* YLM3ST   **********************************
!
! calculate the integral of the product of three real spherical
! harmonics
! i.e    Y_lm Y_l'm' Y_LM
!
! LMAX     max value for l and lp (maximum L is given by triagular rule
!             | l- lp | < L < | l + lp |
!
!*********************************************************************


  SUBROUTINE YLM3ST(LMAX)
      IMPLICIT NONE

      INTEGER LMAX
!f2py intent(in) LMAX

! local variables
      INTEGER S1,S2,S3,T1,T2,T3,IMAX,IC,LMIND,L1,L2,L3,K2,M1,M2,M3, &
              M3P,N1,N2,N3,NM3,I
      REAL(8) Q1,Q2,T,T0
      REAL(8), PARAMETER :: SRPI =1.772453850905516027_8
!---------------------------------------------------------------------
! function to evaluate (-1)^I
!---------------------------------------------------------------------
      IF (LMAXCG>0) RETURN
      LMAXCG=LMAX
!---------------------------------------------------------------------
! set up table for factorials
!---------------------------------------------------------------------
      IMAX=30
      FAC(1)=1._8
      DO I=1,IMAX
         FAC(I+1)= I*FAC(I)
      ENDDO

      IC=0

      LMIND=0
!---------------------------------------------------------------------
! loop over l,m         m =-l,+l
! loop over lp,mp       mp=-lp,+lp
!---------------------------------------------------------------------
      DO L1=0,LMAX
      DO L2=0,LMAX
      K2=(2*L1+1)*(2*L2+1)

      DO M1=-L1,L1
      DO M2=-L2,L2

         LMIND=LMIND+1
         INDCG(LMIND)=IC+1

         N1=IABS(M1)
         S1=0
         IF(M1<0) S1=1
         T1=0
         IF(M1==0) T1=1

         N2=IABS(M2)
         S2=0
         IF(M2<0) S2=1
         T2=0
         IF(M2==0) T2=1

!---------------------------------------------------------------------
! for integrals of 3 real spherical harmonics
! two M values are possibly nonzero
!---------------------------------------------------------------------
         IF(M1*M2<0) THEN
            M3=-N1-N2
            M3P=-IABS(N1-N2)
            IF(M3P==0) THEN
               NM3=1
            ELSE
               NM3=2
            ENDIF
         ELSE IF (M1*M2==0) THEN
            M3=M1+M2
            M3P=0     ! Dummy initialization, not used for this case
            NM3=1
         ELSE
            M3=N1+N2
            M3P=IABS(N1-N2)
            NM3=2
         ENDIF

 5       N3=IABS(M3)
         S3=0
         IF(M3<0) S3=1
         T3=0
         IF(M3==0) T3=1

!---------------------------------------------------------------------
! loop over L given by triangular rule
!---------------------------------------------------------------------
         Q1= 1/2._8*SQRT( REAL(K2,KIND=8))*FS(N3+(S1+S2+S3)/2)
         Q2= 1/(SQRT(2._8)**(1+T1+T2+T3))

         DO L3=ABS(L1-L2),L1+L2, 2

            IF(N3>L3) CYCLE
            T=0._8
            IF(N1+N2==-N3) T=T+CLEBG0(L1,L2,L3)
            IF(N1+N2==N3 ) &
     &           T=T+CLEBGO(L1,L2,L3, N1, N2, N3)*FS(N3+S3)
            IF(N1-N2==-N3) &
     &           T=T+CLEBGO(L1,L2,L3, N1,-N2,-N3)*FS(N2+S2)
            IF(N1-N2==N3 ) &
     &           T=T+CLEBGO(L1,L2,L3,-N1, N2,-N3)*FS(N1+S1)
            IC=IC+1

            IF (IC>NCG)  THEN
               WRITE(*,*) "ERROR: in YLM3ST IC larger than NCG increase NCG and YLM3LOOKUP_TABLE"
               STOP
            ENDIF

            T0=CLEBG0(L1,L2,L3)

            YLM3(IC) = Q1*Q2*T*T0/(SRPI* SQRT( REAL(2*L3+1,KIND=8)))
            IF (T0==0) THEN
               YLM3I(IC)=0
            ELSE
               YLM3I(IC)= T*Q2/Q1/T0*(SRPI* SQRT( REAL(2*L3+1,KIND=8)))
            ENDIF
!           WRITE(*,'(6I4,E14.7)')L3*(L3+1)+M3+1,0,L1,L2,M1,M2,YLM3(IC)

            JL(IC)=L3
            JS(IC)=L3*(L3+1)+M3+1
         ENDDO
! if there is a second M value calculate coefficients for this M
         NM3=NM3-1
         M3=M3P
         IF(NM3>0) GO TO 5

      ENDDO
      ENDDO
      ENDDO
      ENDDO

      INDCG(LMIND+1)=IC+1


      LMIND=0

      DO L1=0,LMAXCG
      DO L2=0,LMAXCG

         YLM3LOOKUP_TABLE(L1,L2)=LMIND
         LMIND=LMIND+(2*L1+1)*(2*L2+1)

      ENDDO
      ENDDO

      RETURN
      
    CONTAINS

      FUNCTION FS(I)
        INTEGER FS,I
        FS=1-2*MOD(I+20,2)
      END FUNCTION FS

  END SUBROUTINE YLM3ST

!************************* YLM3LOOKUP ********************************
!
! function to look up a the startpoint in the array
! YLM3 for two quantumnumbers l lp
!
!*********************************************************************

  SUBROUTINE YLM3LOOKUP(L,LP,LMIND)
      
      IMPLICIT NONE
 
      INTEGER L,LP
      INTEGER LMIND

!f2py intent(in) L,LP
!f2py intent(out) LMIND

      IF (L>LMAXCG .OR. LP>LMAXCG) THEN
         WRITE(*,*)"internal error in YLM3LOOKUP: L or LP > LMAXCG \n L, LP, LMAXCG ", &
            L, " ", LP, " ", LMAXCG
         STOP
      ENDIF
      LMIND=YLM3LOOKUP_TABLE(L,LP)

      RETURN
  END SUBROUTINE YLM3LOOKUP

!************************* SETYLM_GRAD *******************************
!
! calculate spherical harmonics
! and the gradient of the spherical harmonics 
! for a set of grid points up to LMAX 
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
! YLMD(:,1:LMAX,1:3) = d Y_lm(hat x)/d hat x_i
!
! hat x must lie on the unit sphere
!
! to obtain the final gradient one has to use
!
! d Y_lm(x)/ d x_i = sum_j d Y_lm(hat x)/ d hat x_j  d hat x_j/ d x_i
! d Y_lm(x)/ d x_i =
!   sum_j (delta_ij - hat x_i hat x_j)/|x| d Y_lm(hat x)/ d hat x_j 
!
! where hat x = x / ||x||
!
! written by Georg Kresse, modified by Bernhard Schmiedmayer
!
!*********************************************************************

  SUBROUTINE SETYLM_GRAD(YLM,YLMD,INDMAX,INDMAY,LYDIM,R)

      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX,INDMAY        ! number of points (X,Y,Z)
      REAL(8) YLM((LYDIM+1)*(LYDIM+1),INDMAX,INDMAY)        ! spherical harmonics
      REAL(8) YLMD((LYDIM+1)*(LYDIM+1),3,INDMAX,INDMAY)     ! gradient of spherical harmonics
      REAL(8) R(INDMAX,INDMAY,3)  ! x,y and z coordinates


!f2py intent(in) LYDIM,INDMAX,INDMAY,R
!f2py intent(out) YLM,YLMD
!f2py depend(INDMAX) YLMD,YLM,R
!f2py depend(INDMAY) YLMD,YLM,R
!f2py depend(LYDIM) YLMD,YLM

! local variables
      REAL(8), PARAMETER :: PI=3.14159265358979323846_8
      REAL(8) X(INDMAX,INDMAY),Y(INDMAX,INDMAY),Z(INDMAX,INDMAY)
      REAL(8) FAK,YLMX, YLMY, YLMZ
      INTEGER IND,INY,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC,ITMP


     X(:,:) = R(:,:,1)
     Y(:,:) = R(:,:,2)
     Z(:,:) = R(:,:,3)
!-----------------------------------------------------------------------
! runtime check of workspace
!-----------------------------------------------------------------------
      IF ( UBOUND(YLM,1) < (LYDIM+1)**2) THEN
         WRITE(*,*)"internal ERROR: SETYLM_GRAD, insufficient L workspace"
         STOP
      ENDIF
! gfortran compiler bug workaround suggested by jF
      ITMP=UBOUND(YLM,2)
      IF ( ITMP < INDMAX) THEN
!     IF ( UBOUND(YLM,1) < INDMAX) THEN
         WRITE(*,*)"internal ERROR: SETYLM_GRAD, insufficient INDMAX workspace"
         STOP
      ENDIF

      FAK=1/(2._8 * SQRT(PI))
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <0) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)  =FAK
        YLMD(1,:,IND,INY)=0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <1) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(2,IND,INY)  = (FAK*SQRT(3._8))*Y(IND,INY)
        YLM(3,IND,INY)  = (FAK*SQRT(3._8))*Z(IND,INY)
        YLM(4,IND,INY)  = (FAK*SQRT(3._8))*X(IND,INY)
        ! gradient with respect to x
        YLMD(2,1,IND,INY)= 0
        YLMD(3,1,IND,INY)= 0
        YLMD(4,1,IND,INY)= (FAK*SQRT(3._8))
        ! gradient with respect to y
        YLMD(2,2,IND,INY)= (FAK*SQRT(3._8))
        YLMD(3,2,IND,INY)= 0
        YLMD(4,2,IND,INY)= 0
        ! gradient with respect to z
        YLMD(2,3,IND,INY)= 0
        YLMD(3,3,IND,INY)= (FAK*SQRT(3._8))
        YLMD(4,3,IND,INY)= 0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM <2) GOTO 100
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(5,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Y(IND,INY)
        YLM(6,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)*Z(IND,INY)
        YLM(7,IND,INY)= (FAK*SQRT(5._8)/2._8)*(3*Z(IND,INY)*Z(IND,INY)-1)
        YLM(8,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Z(IND,INY)
        YLM(9,IND,INY)= (FAK*SQRT(15._8)/2._8)*(X(IND,INY)*X(IND,INY)-Y(IND,INY)*Y(IND,INY))
        ! gradient with respect to x
        YLMD(5,1,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)
        YLMD(6,1,IND,INY)= 0
        YLMD(7,1,IND,INY)= 0
        YLMD(8,1,IND,INY)= (FAK*SQRT(15._8))  *Z(IND,INY)
        YLMD(9,1,IND,INY)= (FAK*SQRT(15._8)/2._8)*2*X(IND,INY)
        ! gradient with respect to y
        YLMD(5,2,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)
        YLMD(6,2,IND,INY)= (FAK*SQRT(15._8))  *Z(IND,INY)
        YLMD(7,2,IND,INY)= 0
        YLMD(8,2,IND,INY)= 0
        YLMD(9,2,IND,INY)= (FAK*SQRT(15._8)/2._8)*(-2*Y(IND,INY))
        ! gradient with respect to z
        YLMD(5,3,IND,INY)= 0
        YLMD(6,3,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)
        YLMD(7,3,IND,INY)= (FAK*SQRT(5._8)/2._8)*6*Z(IND,INY)
        YLMD(8,3,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)
        YLMD(9,3,IND,INY)= 0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      IF (LYDIM <3) GOTO 100
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(LM,IND,INY) = 0
        YLMD(LM,1,IND,INY) = 0
        YLMD(LM,2,IND,INY) = 0
        YLMD(LM,3,IND,INY) = 0
      ENDDO
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three reel sperical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------
      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)

            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO INY=1,INDMAY
                  DO IND=1,INDMAX
                     YLM(LM,IND,INY) = YLM(LM,IND,INY)+ &
                         YLM3I(IC)*YLM(L*L+M,IND,INY)*YLM(LP*LP+MP,IND,INY)
                     ! gradient
                     YLMD(LM,1,IND,INY) = YLMD(LM,1,IND,INY)+YLM3I(IC)* &
                     (YLMD(L*L+M,1,IND,INY)*YLM(LP*LP+MP,IND,INY)+YLM(L*L+M,IND,INY)*YLMD(LP*LP+MP,1,IND,INY))
                     YLMD(LM,2,IND,INY) = YLMD(LM,2,IND,INY)+YLM3I(IC)* &
                     (YLMD(L*L+M,2,IND,INY)*YLM(LP*LP+MP,IND,INY)+YLM(L*L+M,IND,INY)*YLMD(LP*LP+MP,2,IND,INY))
                     YLMD(LM,3,IND,INY) = YLMD(LM,3,IND,INY)+YLM3I(IC)* &
                     (YLMD(L*L+M,3,IND,INY)*YLM(LP*LP+MP,IND,INY)+YLM(L*L+M,IND,INY)*YLMD(LP*LP+MP,3,IND,INY))
                  ENDDO
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
       ENDDO

 100  CONTINUE
      DO LM=1,(LYDIM+1)*(LYDIM+1)
         DO INY=1,INDMAY
         DO IND=1,INDMAX
            YLMX=YLMD(LM,1,IND,INY)
            YLMY=YLMD(LM,2,IND,INY)
            YLMZ=YLMD(LM,3,IND,INY)

            YLMD(LM,1,IND,INY) = YLMX-X(IND,INY)*X(IND,INY)*YLMX-X(IND,INY)*Y(IND,INY)*YLMY-X(IND,INY)*Z(IND,INY)*YLMZ
            YLMD(LM,2,IND,INY) = YLMY-Y(IND,INY)*X(IND,INY)*YLMX-Y(IND,INY)*Y(IND,INY)*YLMY-Y(IND,INY)*Z(IND,INY)*YLMZ
            YLMD(LM,3,IND,INY) = YLMZ-Z(IND,INY)*X(IND,INY)*YLMX-Z(IND,INY)*Y(IND,INY)*YLMY-Z(IND,INY)*Z(IND,INY)*YLMZ
         ENDDO
         ENDDO
      ENDDO

 END SUBROUTINE SETYLM_GRAD


!*************************** SETYLM **********************************
!
! calculate spherical harmonics 
! for a set of grid points up to LMAX 
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
!
! hat x must lie on the unit sphere
!
! written by Georg Kresse, modified by Bernhard Schmiedmayer
!
!*********************************************************************

  SUBROUTINE SETYLM(YLM,INDMAX,INDMAY,LYDIM,R)

      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX,INDMAY        ! number of points (X,Y,Z)
      REAL(8) YLM((LYDIM+1)*(LYDIM+1),INDMAX,INDMAY)        ! spherical harmonics
      REAL(8) R(INDMAX,INDMAY,3)  ! x,y and z coordinates


!f2py intent(in) LYDIM,INDMAX,INDMAY,R
!f2py intent(out) YLM
!f2py depend(INDMAX) YLM,R
!f2py depend(INDMAY) YLM,R
!f2py depend(LYDIM) YLM

! local variables
      REAL(8), PARAMETER :: PI=3.14159265358979323846_8
      REAL(8) X(INDMAX,INDMAY),Y(INDMAX,INDMAY),Z(INDMAX,INDMAY)
      REAL(8) FAK,YLMX, YLMY, YLMZ
      INTEGER IND,INY,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC,ITMP


     X(:,:) = R(:,:,1)
     Y(:,:) = R(:,:,2)
     Z(:,:) = R(:,:,3)
!-----------------------------------------------------------------------
! runtime check of workspace
!-----------------------------------------------------------------------
      IF ( UBOUND(YLM,1) < (LYDIM+1)**2) THEN
         WRITE(*,*)"internal ERROR: SETYLM, insufficient L workspace"
         STOP
      ENDIF
! gfortran compiler bug workaround suggested by jF
      ITMP=UBOUND(YLM,2)
      IF ( ITMP < INDMAX) THEN
!     IF ( UBOUND(YLM,1) < INDMAX) THEN
         WRITE(*,*)"internal ERROR: SETYLM, insufficient INDMAX workspace"
         STOP
      ENDIF

      FAK=1/(2._8 * SQRT(PI))
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <0) RETURN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)  =FAK
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM <1) RETURN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(2,IND,INY)  = (FAK*SQRT(3._8))*Y(IND,INY)
        YLM(3,IND,INY)  = (FAK*SQRT(3._8))*Z(IND,INY)
        YLM(4,IND,INY)  = (FAK*SQRT(3._8))*X(IND,INY)
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM <2) RETURN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(5,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Y(IND,INY)
        YLM(6,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)*Z(IND,INY)
        YLM(7,IND,INY)= (FAK*SQRT(5._8)/2._8)*(3*Z(IND,INY)*Z(IND,INY)-1)
        YLM(8,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Z(IND,INY)
        YLM(9,IND,INY)= (FAK*SQRT(15._8)/2._8)*(X(IND,INY)*X(IND,INY)-Y(IND,INY)*Y(IND,INY))
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      IF (LYDIM <3) RETURN
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(LM,IND,INY) = 0
      ENDDO
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three reel sperical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------
      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)

            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO INY=1,INDMAY
                  DO IND=1,INDMAX
                     YLM(LM,IND,INY) = YLM(LM,IND,INY)+ &
                         YLM3I(IC)*YLM(L*L+M,IND,INY)*YLM(LP*LP+MP,IND,INY)
                  ENDDO
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
       ENDDO

 END SUBROUTINE SETYLM


!********************** SETYLM_GRAD_LAMB *****************************
!
! calculate spherical harmonics
! and the gradient of the spherical harmonics 
! for a set of grid points for a specific l 
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
! YLMD(:,1:LMAX,1:3) = d Y_lm(hat x)/d hat x_i
!
! hat x must lie on the unit sphere
!
! to obtain the final gradient one has to use
!
! d Y_lm(x)/ d x_i = sum_j d Y_lm(hat x)/ d hat x_j  d hat x_j/ d x_i
! d Y_lm(x)/ d x_i =
!   sum_j (delta_ij - hat x_i hat x_j)/|x| d Y_lm(hat x)/ d hat x_j 
!
! where hat x = x / ||x||
!
! written by Georg Kresse, modified by Bernhard Schmiedmayer
!
!*********************************************************************

  SUBROUTINE SETYLM_GRAD_LAMB(YLM,YLMD,INDMAX,INDMAY,LYDIM,R)

      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX,INDMAY        ! number of points (X,Y,Z)
      REAL(8) YLM((2*LYDIM+1),INDMAX,INDMAY)        ! spherical harmonics
      REAL(8) YLMD((2*LYDIM+1),3,INDMAX,INDMAY)     ! gradient of spherical harmonics
      REAL(8) R(INDMAX,INDMAY,3)  ! x,y and z coordinates


!f2py intent(in) LYDIM,INDMAX,INDMAY,R
!f2py intent(out) YLM,YLMD
!f2py depend(INDMAX) YLMD,YLM,R
!f2py depend(INDMAY) YLMD,YLM,R
!f2py depend(LYDIM) YLMD,YLM

! local variables
      REAL(8), PARAMETER :: PI=3.14159265358979323846_8
      REAL(8) X(INDMAX,INDMAY),Y(INDMAX,INDMAY),Z(INDMAX,INDMAY)
      REAL(8) FAK,YLMX, YLMY, YLMZ
      INTEGER IND,INY,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC,ITMP
      REAL(8), ALLOCATABLE :: GYLM(:,:,:)        ! spherical harmonics
      REAL(8), ALLOCATABLE :: GYLMD(:,:,:,:)     ! gradient of spherical harmonics

     X(:,:) = R(:,:,1)
     Y(:,:) = R(:,:,2)
     Z(:,:) = R(:,:,3)
!-----------------------------------------------------------------------
! runtime check of workspace
!-----------------------------------------------------------------------
      IF ( UBOUND(YLM,1) < (2*LYDIM+1)) THEN
         WRITE(*,*)"internal ERROR: SETYLM_GRAD_LAMB, insufficient L workspace"
         STOP
      ENDIF
! gfortran compiler bug workaround suggested by jF
      ITMP=UBOUND(YLM,2)
      IF ( ITMP < INDMAX) THEN
!     IF ( UBOUND(YLM,1) < INDMAX) THEN
         WRITE(*,*)"internal ERROR: SETYLM_GRAD_LAMB, insufficient INDMAX workspace"
         STOP
      ENDIF

      FAK=1/(2._8 * SQRT(PI))
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM == 0) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)  =FAK
        YLMD(1,:,IND,INY)=0
      ENDDO
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM == 1) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)  = (FAK*SQRT(3._8))*Y(IND,INY)
        YLM(2,IND,INY)  = (FAK*SQRT(3._8))*Z(IND,INY)
        YLM(3,IND,INY)  = (FAK*SQRT(3._8))*X(IND,INY)
        ! gradient with respect to x
        YLMD(1,1,IND,INY)= 0
        YLMD(2,1,IND,INY)= 0
        YLMD(3,1,IND,INY)= (FAK*SQRT(3._8))
        ! gradient with respect to y
        YLMD(1,2,IND,INY)= (FAK*SQRT(3._8))
        YLMD(2,2,IND,INY)= 0
        YLMD(3,2,IND,INY)= 0
        ! gradient with respect to z
        YLMD(1,3,IND,INY)= 0
        YLMD(2,3,IND,INY)= (FAK*SQRT(3._8))
        YLMD(3,3,IND,INY)= 0
      ENDDO
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM == 2) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Y(IND,INY)
        YLM(2,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)*Z(IND,INY)
        YLM(3,IND,INY)= (FAK*SQRT(5._8)/2._8)*(3*Z(IND,INY)*Z(IND,INY)-1)
        YLM(4,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Z(IND,INY)
        YLM(5,IND,INY)= (FAK*SQRT(15._8)/2._8)*(X(IND,INY)*X(IND,INY)-Y(IND,INY)*Y(IND,INY))
        ! gradient with respect to x
        YLMD(1,1,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)
        YLMD(2,1,IND,INY)= 0
        YLMD(3,1,IND,INY)= 0
        YLMD(4,1,IND,INY)= (FAK*SQRT(15._8))  *Z(IND,INY)
        YLMD(5,1,IND,INY)= (FAK*SQRT(15._8)/2._8)*2*X(IND,INY)
        ! gradient with respect to y
        YLMD(1,2,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)
        YLMD(2,2,IND,INY)= (FAK*SQRT(15._8))  *Z(IND,INY)
        YLMD(3,2,IND,INY)= 0
        YLMD(4,2,IND,INY)= 0
        YLMD(5,2,IND,INY)= (FAK*SQRT(15._8)/2._8)*(-2*Y(IND,INY))
        ! gradient with respect to z
        YLMD(1,3,IND,INY)= 0
        YLMD(2,3,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)
        YLMD(3,3,IND,INY)= (FAK*SQRT(5._8)/2._8)*6*Z(IND,INY)
        YLMD(4,3,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)
        YLMD(5,3,IND,INY)= 0
      ENDDO
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! code for L=3,
!-----------------------------------------------------------------------
      IF (LYDIM == 3) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)= (FAK*SQRT(17.5_8)/2._8) *Y(IND,INY)*(3._8*X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))
        YLM(2,IND,INY)= (FAK*SQRT(105._8)) *X(IND,INY)*Y(IND,INY)*Z(IND,INY)
        YLM(3,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *Y(IND,INY)*(5._8*Z(IND,INY)*Z(IND,INY)-1._8)
        YLM(4,IND,INY)= (FAK*SQRT(7._8)/2._8)  *(5._8*Z(IND,INY)*Z(IND,INY) - 3._8)*Z(IND,INY) 
        YLM(5,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *X(IND,INY)*(5._8*Z(IND,INY)*Z(IND,INY)-1._8)
        YLM(6,IND,INY)= (FAK*SQRT(105._8)/2._8) *(X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))*Z(IND,INY)
        YLM(7,IND,INY)= (FAK*SQRT(17.5_8)/2._8) *X(IND,INY)*(X(IND,INY)*X(IND,INY) - 3._8*Y(IND,INY)*Y(IND,INY))
        ! gradient with respect to x
        YLMD(1,1,IND,INY)= (FAK*SQRT(17.5_8)) *3._8*X(IND,INY)*Y(IND,INY)
        YLMD(2,1,IND,INY)= (FAK*SQRT(105._8)) *Y(IND,INY)*Z(IND,INY)
        YLMD(3,1,IND,INY)= 0._8
        YLMD(4,1,IND,INY)= 0._8
        YLMD(5,1,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *(5._8*Z(IND,INY)*Z(IND,INY)-1._8)
        YLMD(6,1,IND,INY)= (FAK*SQRT(105._8)) *X(IND,INY)*Z(IND,INY)
        YLMD(7,1,IND,INY)= (FAK*SQRT(17.5_8)/2._8) *3._8*(X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))
        ! gradient with respect to y
        YLMD(1,2,IND,INY)= (FAK*SQRT(17.5_8)/2._8) *3._8*(X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))
        YLMD(2,2,IND,INY)= (FAK*SQRT(105._8)) *X(IND,INY)*Z(IND,INY)
        YLMD(3,2,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *(5._8*Z(IND,INY)*Z(IND,INY)-1._8)
        YLMD(4,2,IND,INY)= 0._8 
        YLMD(5,2,IND,INY)= 0._8
        YLMD(6,2,IND,INY)= (FAK*SQRT(105._8)) *(-Y(IND,INY)*Z(IND,INY))
        YLMD(7,2,IND,INY)= (FAK*SQRT(17.5_8)) * (-3._8*X(IND,INY)*Y(IND,INY))
        ! gradient with respect to z
        YLMD(1,3,IND,INY)= 0._8
        YLMD(2,3,IND,INY)= (FAK*SQRT(105._8)) *X(IND,INY)*Y(IND,INY)
        YLMD(3,3,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *10._8*Y(IND,INY)*Z(IND,INY)
        YLMD(4,3,IND,INY)= (FAK*SQRT(7._8)/2._8)  *(15._8*Z(IND,INY)*Z(IND,INY) - 3._8) 
        YLMD(5,3,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *10._8*X(IND,INY)*Z(IND,INY)
        YLMD(6,3,IND,INY)= (FAK*SQRT(105._8)/2._8) *(X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))
        YLMD(7,3,IND,INY)= 0._8
      ENDDO
      ENDDO
      ENDIF

      IF (LYDIM > 3) THEN
      ALLOCATE(GYLM((LYDIM+1)*(LYDIM+1),INDMAX,INDMAY))        ! spherical harmonics
      ALLOCATE(GYLMD((LYDIM+1)*(LYDIM+1),3,INDMAX,INDMAY))     ! gradient of spherical harmonics

!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        GYLM(1,IND,INY)  = FAK
        GYLM(2,IND,INY)  = (FAK*SQRT(3._8))*Y(IND,INY)
        GYLM(3,IND,INY)  = (FAK*SQRT(3._8))*Z(IND,INY)
        GYLM(4,IND,INY)  = (FAK*SQRT(3._8))*X(IND,INY)
        GYLM(5,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Y(IND,INY)
        GYLM(6,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)*Z(IND,INY)
        GYLM(7,IND,INY)= (FAK*SQRT(5._8)/2._8)*(3*Z(IND,INY)*Z(IND,INY)-1)
        GYLM(8,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Z(IND,INY)
        GYLM(9,IND,INY)= (FAK*SQRT(15._8)/2._8)*(X(IND,INY)*X(IND,INY)-Y(IND,INY)*Y(IND,INY))
        ! gradient with respect to x
        GYLMD(1,1,IND,INY)= 0
        GYLMD(2,1,IND,INY)= 0
        GYLMD(3,1,IND,INY)= 0
        GYLMD(4,1,IND,INY)= (FAK*SQRT(3._8))
        GYLMD(5,1,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)
        GYLMD(6,1,IND,INY)= 0
        GYLMD(7,1,IND,INY)= 0
        GYLMD(8,1,IND,INY)= (FAK*SQRT(15._8))  *Z(IND,INY)
        GYLMD(9,1,IND,INY)= (FAK*SQRT(15._8)/2._8)*2*X(IND,INY)
        ! gradient with respect to y
        GYLMD(1,2,IND,INY)= 0
        GYLMD(2,2,IND,INY)= (FAK*SQRT(3._8))
        GYLMD(3,2,IND,INY)= 0
        GYLMD(4,2,IND,INY)= 0
        GYLMD(5,2,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)
        GYLMD(6,2,IND,INY)= (FAK*SQRT(15._8))  *Z(IND,INY)
        GYLMD(7,2,IND,INY)= 0
        GYLMD(8,2,IND,INY)= 0
        GYLMD(9,2,IND,INY)= (FAK*SQRT(15._8)/2._8)*(-2*Y(IND,INY))
        ! gradient with respect to z
        GYLMD(1,3,IND,INY)= 0
        GYLMD(2,3,IND,INY)= 0
        GYLMD(3,3,IND,INY)= (FAK*SQRT(3._8))
        GYLMD(4,3,IND,INY)= 0
        GYLMD(5,3,IND,INY)= 0
        GYLMD(6,3,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)
        GYLMD(7,3,IND,INY)= (FAK*SQRT(5._8)/2._8)*6*Z(IND,INY)
        GYLMD(8,3,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)
        GYLMD(9,3,IND,INY)= 0
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        GYLM(LM,IND,INY) = 0
        GYLMD(LM,1,IND,INY) = 0
        GYLMD(LM,2,IND,INY) = 0
        GYLMD(LM,3,IND,INY) = 0
      ENDDO
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three reel sperical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------
      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)

            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO INY=1,INDMAY
                  DO IND=1,INDMAX
                     GYLM(LM,IND,INY) = GYLM(LM,IND,INY)+ &
                         YLM3I(IC)*GYLM(L*L+M,IND,INY)*GYLM(LP*LP+MP,IND,INY)
                     ! gradient
                     GYLMD(LM,1,IND,INY) = GYLMD(LM,1,IND,INY)+YLM3I(IC)* &
                     (GYLMD(L*L+M,1,IND,INY)*GYLM(LP*LP+MP,IND,INY)+GYLM(L*L+M,IND,INY)*GYLMD(LP*LP+MP,1,IND,INY))
                     GYLMD(LM,2,IND,INY) = GYLMD(LM,2,IND,INY)+YLM3I(IC)* &
                     (GYLMD(L*L+M,2,IND,INY)*GYLM(LP*LP+MP,IND,INY)+GYLM(L*L+M,IND,INY)*GYLMD(LP*LP+MP,2,IND,INY))
                     GYLMD(LM,3,IND,INY) = GYLMD(LM,3,IND,INY)+YLM3I(IC)* &
                     (GYLMD(L*L+M,3,IND,INY)*GYLM(LP*LP+MP,IND,INY)+GYLM(L*L+M,IND,INY)*GYLMD(LP*LP+MP,3,IND,INY))
                  ENDDO
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
       ENDDO

      LM = (LYDIM) * (LYDIM)
      DO L=1, (2*LYDIM + 1)
        LM = LM + 1
!DIR$ IVDEP
!OCL NOVREC
        DO INY=1,INDMAY
        DO IND=1,INDMAX
          YLM(L,IND,INY) = GYLM(LM,IND,INY)
          ! gradient
          YLMD(L,1,IND,INY) = GYLMD(LM,1,IND,INY)
          YLMD(L,2,IND,INY) = GYLMD(LM,2,IND,INY)
          YLMD(L,3,IND,INY) = GYLMD(LM,3,IND,INY)
        ENDDO
        ENDDO
      ENDDO
      ENDIF

      DO LM=1,(2*LYDIM+1)
!DIR$ IVDEP
!OCL NOVREC
         DO INY=1,INDMAY
         DO IND=1,INDMAX
            YLMX=YLMD(LM,1,IND,INY)
            YLMY=YLMD(LM,2,IND,INY)
            YLMZ=YLMD(LM,3,IND,INY)

            YLMD(LM,1,IND,INY) = YLMX-X(IND,INY)*X(IND,INY)*YLMX-X(IND,INY)*Y(IND,INY)*YLMY-X(IND,INY)*Z(IND,INY)*YLMZ
            YLMD(LM,2,IND,INY) = YLMY-Y(IND,INY)*X(IND,INY)*YLMX-Y(IND,INY)*Y(IND,INY)*YLMY-Y(IND,INY)*Z(IND,INY)*YLMZ
            YLMD(LM,3,IND,INY) = YLMZ-Z(IND,INY)*X(IND,INY)*YLMX-Z(IND,INY)*Y(IND,INY)*YLMY-Z(IND,INY)*Z(IND,INY)*YLMZ
         ENDDO
         ENDDO
      ENDDO

 END SUBROUTINE SETYLM_GRAD_LAMB

!************************* SETYLM_LAMB *******************************
!
! calculate spherical harmonics 
! for a set of grid points for a specific l 
!
! YLM(:,1:LMAX)      = Y_lm(hat x)
!
! hat x must lie on the unit sphere
!
! written by Georg Kresse, modified by Bernhard Schmiedmayer
!
!*********************************************************************

  SUBROUTINE SETYLM_LAMB(YLM,INDMAX,INDMAY,LYDIM,R)

      IMPLICIT NONE
      INTEGER LYDIM           ! maximum L
      INTEGER INDMAX,INDMAY        ! number of points (X,Y,Z)
      REAL(8) YLM((2*LYDIM+1),INDMAX,INDMAY)        ! spherical harmonics
      REAL(8) R(INDMAX,INDMAY,3)  ! x,y and z coordinates


!f2py intent(in) LYDIM,INDMAX,INDMAY,R
!f2py intent(out) YLM
!f2py depend(INDMAX) YLM,R
!f2py depend(INDMAY) YLM,R
!f2py depend(LYDIM) YLM

! local variables
      REAL(8), PARAMETER :: PI=3.14159265358979323846_8
      REAL(8) X(INDMAX,INDMAY),Y(INDMAX,INDMAY),Z(INDMAX,INDMAY)
      REAL(8) FAK,YLMX, YLMY, YLMZ
      INTEGER IND,INY,LSET,LM,LP,LMINDX,ISTART,IEND,LNEW,L,M,MP,IC,ITMP
      REAL(8), ALLOCATABLE :: GYLM(:,:,:)        ! spherical harmonics

     X(:,:) = R(:,:,1)
     Y(:,:) = R(:,:,2)
     Z(:,:) = R(:,:,3)
!-----------------------------------------------------------------------
! runtime check of workspace
!-----------------------------------------------------------------------
      IF ( UBOUND(YLM,1) < (2*LYDIM+1)) THEN
         WRITE(*,*)"internal ERROR: SETYLM_GRAD_LAMB, insufficient L workspace"
         STOP
      ENDIF
! gfortran compiler bug workaround suggested by jF
      ITMP=UBOUND(YLM,2)
      IF ( ITMP < INDMAX) THEN
!     IF ( UBOUND(YLM,1) < INDMAX) THEN
         WRITE(*,*)"internal ERROR: SETYLM_GRAD_LAMB, insufficient INDMAX workspace"
         STOP
      ENDIF

      FAK=1/(2._8 * SQRT(PI))
!-----------------------------------------------------------------------
! here is the code for L=0, hard coded
!-----------------------------------------------------------------------
      IF (LYDIM == 0) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)  =FAK
      ENDDO
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! here is the code for L=1, once again hard coded
!-----------------------------------------------------------------------
      IF (LYDIM == 1) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)  = (FAK*SQRT(3._8))*Y(IND,INY)
        YLM(2,IND,INY)  = (FAK*SQRT(3._8))*Z(IND,INY)
        YLM(3,IND,INY)  = (FAK*SQRT(3._8))*X(IND,INY)
      ENDDO
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! code for L=2,
!-----------------------------------------------------------------------
      IF (LYDIM == 2) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Y(IND,INY)
        YLM(2,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)*Z(IND,INY)
        YLM(3,IND,INY)= (FAK*SQRT(5._8)/2._8)*(3*Z(IND,INY)*Z(IND,INY)-1)
        YLM(4,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Z(IND,INY)
        YLM(5,IND,INY)= (FAK*SQRT(15._8)/2._8)*(X(IND,INY)*X(IND,INY)-Y(IND,INY)*Y(IND,INY))
      ENDDO
      ENDDO
      ENDIF
!-----------------------------------------------------------------------
! code for L=3,
!-----------------------------------------------------------------------
      IF (LYDIM == 3) THEN
!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        YLM(1,IND,INY)= (FAK*SQRT(17.5_8)/2._8) *Y(IND,INY)*(3._8*X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))
        YLM(2,IND,INY)= (FAK*SQRT(105._8)) *X(IND,INY)*Y(IND,INY)*Z(IND,INY)
        YLM(3,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *Y(IND,INY)*(5._8*Z(IND,INY)*Z(IND,INY)-1._8)
        YLM(4,IND,INY)= (FAK*SQRT(7._8)/2._8)  *(5._8*Z(IND,INY)*Z(IND,INY) - 3._8)*Z(IND,INY) 
        YLM(5,IND,INY)= (FAK*SQRT(10.5_8)/2._8) *X(IND,INY)*(5._8*Z(IND,INY)*Z(IND,INY)-1._8)
        YLM(6,IND,INY)= (FAK*SQRT(105._8)/2._8) *(X(IND,INY)*X(IND,INY) - Y(IND,INY)*Y(IND,INY))*Z(IND,INY)
        YLM(7,IND,INY)= (FAK*SQRT(17.5_8)/2._8) *X(IND,INY)*(X(IND,INY)*X(IND,INY) - 3._8*Y(IND,INY)*Y(IND,INY))
      ENDDO
      ENDDO
      ENDIF

      IF (LYDIM > 3) THEN
      ALLOCATE(GYLM((LYDIM+1)*(LYDIM+1),INDMAX,INDMAY))        ! spherical harmonics

!DIR$ IVDEP
!OCL NOVREC
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        GYLM(1,IND,INY)  = FAK
        GYLM(2,IND,INY)  = (FAK*SQRT(3._8))*Y(IND,INY)
        GYLM(3,IND,INY)  = (FAK*SQRT(3._8))*Z(IND,INY)
        GYLM(4,IND,INY)  = (FAK*SQRT(3._8))*X(IND,INY)
        GYLM(5,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Y(IND,INY)
        GYLM(6,IND,INY)= (FAK*SQRT(15._8))  *Y(IND,INY)*Z(IND,INY)
        GYLM(7,IND,INY)= (FAK*SQRT(5._8)/2._8)*(3*Z(IND,INY)*Z(IND,INY)-1)
        GYLM(8,IND,INY)= (FAK*SQRT(15._8))  *X(IND,INY)*Z(IND,INY)
        GYLM(9,IND,INY)= (FAK*SQRT(15._8)/2._8)*(X(IND,INY)*X(IND,INY)-Y(IND,INY)*Y(IND,INY))
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! initialize all componentes L>2 to zero
!-----------------------------------------------------------------------
      LSET=2

      DO LM=(LSET+1)*(LSET+1)+1,(LYDIM+1)*(LYDIM+1)
      DO INY=1,INDMAY
      DO IND=1,INDMAX
        GYLM(LM,IND,INY) = 0
      ENDDO
      ENDDO
      ENDDO
!-----------------------------------------------------------------------
! for L>2 we use (some kind of) Clebsch-Gordan coefficients
! i.e. the inverse of the integral of three reel sperical harmonics
!      Y_LM = \sum_ll'mm'  C_ll'mm'(L,M) Y_lm Y_l'm'
!-----------------------------------------------------------------------
      LP=1
      DO L=LSET,LYDIM-1
         CALL YLM3LOOKUP(L,LP,LMINDX)
         LNEW=L+LP
         DO M = 1, 2*L +1
         DO MP= 1, 2*LP+1
            LMINDX=LMINDX+1

            ISTART=INDCG(LMINDX)
            IEND  =INDCG(LMINDX+1)

            DO IC=ISTART,IEND-1
               LM=JS(IC)
               IF (LM > LNEW*LNEW       .AND. &
                   LM <= (LNEW+1)*(LNEW+1)) THEN
!DIR$ IVDEP
!OCL NOVREC
!                   IF (LNEW == 2) THEN
!                      WRITE(*,*)LNEW,LM,L*L+M,LP*LP+MP,YLM3I(IC)
!                   ENDIF
                  DO INY=1,INDMAY
                  DO IND=1,INDMAX
                     GYLM(LM,IND,INY) = GYLM(LM,IND,INY)+ &
                         YLM3I(IC)*GYLM(L*L+M,IND,INY)*GYLM(LP*LP+MP,IND,INY)
                  ENDDO
                  ENDDO
               ENDIF
            ENDDO
         ENDDO
         ENDDO
       ENDDO

      LM = (LYDIM) * (LYDIM)
      DO L=1, (2*LYDIM + 1)
        LM = LM + 1
!DIR$ IVDEP
!OCL NOVREC
        DO INY=1,INDMAY
        DO IND=1,INDMAX
          YLM(L,IND,INY) = GYLM(LM,IND,INY)
        ENDDO
        ENDDO
      ENDDO
      ENDIF

 END SUBROUTINE SETYLM_LAMB

END MODULE asa
