!     Last change:  PND   6 Sep 2007    5:56 pm

module parameters
implicit none
   REAL, PARAMETER  	:: b = 0.99, d = 0.025, a = 0.36
   REAL, PARAMETER  	:: klb = 0.01, inc = 0.075, kub = 75.0
   INTEGER, PARAMETER 	:: length_grid_k = (kub-klb)/inc + 1
   REAL , PARAMETER 	:: toler   = 1.e-4						! Numerical tolerance
   REAL, PARAMETER      :: pi_h_h = 0.977, pi_h_l = 0.023, pi_l_h = 0.074, pi_l_l=0.926
   REAL, PARAMETER      :: z_h = 1.25, z_l = 0.20
end module

! ============================================================================================
module global
USE parameters
implicit none
   REAL 		:: Kgrid(length_grid_k)
   REAL                 :: value_h(length_grid_k), value_l(length_grid_k)
   REAL                 :: g_k_h(length_grid_k), g_k_l(length_grid_k)
   REAL                 :: vtmp_h(length_grid_k,length_grid_k), vtmp_l(length_grid_k,length_grid_k)
   REAL                 :: value_new_h(length_grid_k), value_new_l(length_grid_k)
end module

! ============================================================================================

PROGRAM  HW2NonStochastic
   REAL 			:: total, etime, dist
   REAL, DIMENSION(2)  		:: elapsed

   call solution

   total=etime(elapsed)


	PRINT*,'--------------------------------------------------'
	PRINT*,'total time elpased =',total
	PRINT*,'--------------------------------------------------'


END PROGRAM HW2NonStochastic

! ============================================================================================
subroutine solution
   USE parameters
   USE global

   IMPLICIT  NONE

   INTEGER :: iter, index_k, index_kp
   REAL :: diff, diff_h, diff_l, k, kp, c_l, c_h
   
   INTEGER :: i = 1


   do while (i<=length_grid_k)   !do loop for assigning capital grid K
     Kgrid(i) = klb + (i-1)*inc
     !write(*,*) i, Kgrid(i)
     i = i + 1
   end do


   iter = 1
   diff = 1000.d0
   value_h = 0.*Kgrid		!Initial Value guess
   value_l = 0.*Kgrid

	do while (diff>= toler)

		do index_k = 1, length_grid_k				! Capital grid
			k = Kgrid(index_k)
                        vtmp_h(index_k,:) = -1.0e-16
                        vtmp_l(index_k, :) = -1.0e-16

			do index_kp = 1, length_grid_k
				kp = Kgrid(index_kp)
				c_h = z_h * k**a+(1.-d)*k-kp
                                c_l = z_l * k**a+(1.-d)*k-kp

				if (c_h>0.) then
	                        	vtmp_h(index_k,index_kp) = log(c_h)+b*(pi_h_h*value_h(index_kp) + pi_h_l*value_l(index_kp))
	          		endif
                                if (c_l>0.) then
	                        	vtmp_l(index_k,index_kp) = log(c_l)+b*(pi_l_h*value_h(index_kp) + pi_l_l*value_l(index_kp))
	          		endif

			enddo

			value_new_h(index_k) = MAXVAL(vtmp_h(index_k,:))
                        value_new_l(index_k) = MAXVAL(vtmp_l(index_k,:))
                        g_k_h(index_k) = Kgrid(MAXLOC(vtmp_h(index_k,:),1))
                        g_k_l(index_k) = Kgrid(MAXLOC(vtmp_l(index_k,:),1))
                        
                enddo

		diff_h  = maxval(abs(value_new_h-value_h))/ABS(value_new_h(length_grid_k))
		diff_l = maxval(abs(value_new_l-value_l))/ABS(value_new_l(length_grid_k))
                diff = max(diff_h, diff_l)
                value_h = value_new_h
                value_l = value_new_l


		print*, 'Iteration =',iter,'sup_norm =',diff
                iter = iter+1

	enddo

	print *, ' '
	print *, 'Successfully converged with sup_norm ', diff
    	!print *, g_k
    
    !CALL vcDrawCurve@(d, Kgrid, g_k, length_grid_k)


        open (UNIT=1,FILE='valuefun.csv',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) Kgrid(index_k), ",", value_h(index_k), ",", value_l(index_k)
        end do
	close (UNIT=1)

        open (UNIT=1,FILE='polfun.csv',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) Kgrid(index_k), ",", g_k_h(index_k), ",", g_k_l(index_k)
        end do
	close (UNIT=1)
        

end subroutine
