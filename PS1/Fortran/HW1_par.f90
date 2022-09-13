! Jake Zhao
! Department of Economics
! University of Wisconsin-Madison
! Econ 899: Computation of Heterogeneous Agent Models (Dean Corbae)
! Fall 2012

! Non stochastic growth model with MPI
! Compile with the command "mpif90 HW2ns_parallel.f90 -o hw2 -O2"
! Run with the command "mpirun -np 2 hw2"

PROGRAM HW2NonStochastic
IMPLICIT NONE
INCLUDE "mpif.h"

	REAL,    PARAMETER                :: b = 0.99, d = 0.025, a = 0.36
	REAL,    PARAMETER                :: klb = 0.01, inc = 0.075, kub = 75.0
	REAL,	 PARAMETER				  :: pi_h_h = 0.977, pi_h_l = 0.023, pi_l_h = 0.074, pi_l_l = 0.926
	REAL,    PARAMETER                :: toler = 1e-4
	INTEGER, PARAMETER                :: length_grid_k = (kub-klb)/inc+1
	REAL,    DIMENSION(length_grid_k) :: Kgrid, value_h, value_l, g_k_l, g_k_h, value_prov_h, value_prov_l, g_k_prov_h, g_k_prov_l
	REAL                              :: diff, diff_h, diff_l, k, kp, c_h, c_l, value_max_so_far_h, value_max_so_far_l, &
										value_comp_h, value_comp_l
	INTEGER                           :: iter, go_on, index_k, index_kp, index_maxk_so_far_l, index_maxk_so_far_h
	INTEGER                           :: ierror          ! Error message from MPI subroutines
	INTEGER                           :: proc_id         ! Identification number of each processor
	INTEGER                           :: num_proc        ! Total number of processors
	INTEGER                           :: master_id = 0   ! Master processor identification number

	REAL, DIMENSION(:), ALLOCATABLE   :: value_prov_temp_h, value_prov_temp_l, g_k_prov_temp_h, g_k_prov_temp_l

	INTEGER :: i = 1

	REAL    :: total, etime, dist
	REAL    :: elapsed(2)

	CALL MPI_init(ierror)                                ! Starts MPI processes
	CALL MPI_comm_rank(MPI_comm_world, proc_id, ierror)  ! Find the process ranks for each process, proc_id = 0, 1, 2, ...
	CALL MPI_comm_size(MPI_comm_world, num_proc, ierror) ! Find number of processes requested, determined when using mpirun

	DO WHILE (i <= length_grid_k) ! Do loop for assigning capital grid K
		Kgrid(i) = klb + (i-1)*inc
		i = i+1
	END DO

	iter  = 1
	diff_h  = REAL(1000)
	diff_l  = REAL(1000)
	diff  = REAL(1000)
	value_h = REAL(0)
	value_l = REAL(0)

	ALLOCATE(value_prov_temp_h(length_grid_k/num_proc))
	ALLOCATE(value_prov_temp_l(length_grid_k/num_proc))
	ALLOCATE(g_k_prov_temp_h(length_grid_k/num_proc))
	ALLOCATE(g_k_prov_temp_l(length_grid_k/num_proc))

	go_on = 1

	DO WHILE (go_on == 1)
		IF (proc_id == master_id) THEN
			iter = iter+1
		END IF

		index_maxk_so_far_l = 1
		value_prov_temp_l   = 0
		g_k_prov_temp_l     = 0
		index_maxk_so_far_h = 1
		value_prov_temp_h   = 0
		g_k_prov_temp_h     = 0
		

		DO index_k = 1, length_grid_k/num_proc ! Capital grid
			k = Kgrid(index_k+proc_id*length_grid_k/num_proc)
			value_max_so_far_h = REAL(-100000000)
			value_max_so_far_l = REAL(-100000000)

			DO index_kp = index_maxk_so_far_h, length_grid_k
				kp = Kgrid(index_kp)
				c_h = 1.25*(k**a)+(1.-d)*k-kp

				IF (c_h > 0) THEN
					value_comp_h = log(c_h)+b*(pi_h_h*value_h(index_kp) + pi_h_l*value_l(index_kp))
					IF (value_comp_h > value_max_so_far_h) THEN
						value_max_so_far_h = value_comp_h
						g_k_prov_temp_h(index_k) = kp
						index_maxk_so_far_h = index_kp
					END IF
				ELSE
					EXIT
				END IF
			END DO
			DO index_kp = index_maxk_so_far_l, length_grid_k
				kp = Kgrid(index_kp)
				c_l = 0.2*(k**a)+(1.-d)*k-kp

				IF (c_l > 0) THEN
					value_comp_l = log(c_l)+b*(pi_l_h*value_h(index_kp) + pi_l_l*value_l(index_kp))
					IF (value_comp_l > value_max_so_far_l) THEN
						value_max_so_far_l = value_comp_l
						g_k_prov_temp_l(index_k) = kp
						index_maxk_so_far_l = index_kp
					END IF
				ELSE
					EXIT
				END IF
			END DO

			value_prov_temp_h(index_k) = value_max_so_far_h
			value_prov_temp_l(index_k) = value_max_so_far_l
		END DO

		CALL MPI_barrier(MPI_comm_world, ierror)
		CALL MPI_gather(value_prov_temp_h,&                                                        ! the vectors to be gathered
		    & length_grid_k/num_proc,&                                                           ! the number of elements to be gathered
		    & MPI_real,&                                                                         ! the element type of vectors to be gathered
		    & value_prov_h(proc_id*length_grid_k/num_proc+1:(proc_id+1)*length_grid_k/num_proc),&  ! the gathered location
			& length_grid_k/num_proc,&                                                           ! the number of elements to be gathered
			& MPI_real,&                                                                         ! the element type of gathered location
			& master_id,&                                                                        ! the processor that does the gathering
			& MPI_comm_world,&                                                                   ! the scope
			& ierror)   
		CALL MPI_gather(value_prov_temp_l,&                                                        ! the vectors to be gathered
		    & length_grid_k/num_proc,&                                                           ! the number of elements to be gathered
		    & MPI_real,&                                                                         ! the element type of vectors to be gathered
		    & value_prov_l(proc_id*length_grid_k/num_proc+1:(proc_id+1)*length_grid_k/num_proc),&  ! the gathered location
			& length_grid_k/num_proc,&                                                           ! the number of elements to be gathered
			& MPI_real,&                                                                         ! the element type of gathered location
			& master_id,&                                                                        ! the processor that does the gathering
			& MPI_comm_world,&                                                                   ! the scope
			& ierror)                                                                            ! the return value of a possible error
		CALL MPI_gather(g_k_prov_temp_h, &
		    & length_grid_k/num_proc,&
		    & MPI_real,&
		    & g_k_prov_h(proc_id*length_grid_k/num_proc+1:(proc_id+1)*length_grid_k/num_proc), &
			& length_grid_k/num_proc,&
			& MPI_real,&
			& master_id, &
			& MPI_comm_world,&
			& ierror)
		CALL MPI_gather(g_k_prov_temp_l, &
		    & length_grid_k/num_proc,&
		    & MPI_real,&
		    & g_k_prov_l(proc_id*length_grid_k/num_proc+1:(proc_id+1)*length_grid_k/num_proc), &
			& length_grid_k/num_proc,&
			& MPI_real,&
			& master_id, &
			& MPI_comm_world,&
			& ierror)
		CALL MPI_bcast(value_prov_h,&  ! the vectors to be broadcasted
		    & length_grid_k,&        ! the number of elemebts to be broadcasted
		    & MPI_real,&             ! the elemet type of vectors to be broadcasted
		    & master_id,&            ! the processor that does the gathering
		    & MPI_comm_world,&       ! the scope
		    & ierror)                ! the return value of a possible error
		CALL MPI_bcast(value_prov_l,&  ! the vectors to be broadcasted
		    & length_grid_k,&        ! the number of elemebts to be broadcasted
		    & MPI_real,&             ! the elemet type of vectors to be broadcasted
		    & master_id,&            ! the processor that does the gathering
		    & MPI_comm_world,&       ! the scope
		    & ierror)                ! the return value of a possible error
		CALL MPI_bcast(g_k_prov_h, &
		    & length_grid_k,&
		    & MPI_real,&
		    & master_id,&
		    & MPI_comm_world,&
		    & ierror)
		CALL MPI_bcast(g_k_prov_l, &
		    & length_grid_k,&
		    & MPI_real,&
		    & master_id,&
		    & MPI_comm_world,&
		    & ierror)

		IF (proc_id == master_id)THEN
			diff_h = maxval(abs(value_prov_h-value_h))/abs(value_prov_h(length_grid_k))
			diff_l = maxval(abs(value_prov_l-value_l))/abs(value_prov_l(length_grid_k))
			diff = max(diff_h, diff_l)

			IF (diff < toler) THEN
				go_on = 0
			END IF

			PRINT*, 'Iteration =', iter-1, 'sup_norm =', diff
		END IF

		value_h = value_prov_h
		value_l = value_prov_l
		g_k_h   = g_k_prov_h
		g_k_l   = g_k_prov_l

		CALL MPI_bcast(go_on, 1, MPI_integer, master_id, MPI_comm_world, ierror)
	END DO

	IF (proc_id == master_id) THEN
		PRINT*, ''
		PRINT*, 'Successfully converged with sup_norm ', diff

		total = etime(elapsed)
		PRINT*, '--------------------------------------------------'
		PRINT*, 'Total time elapsed =', total
		PRINT*, '--------------------------------------------------'
	END IF

	CALL MPI_finalize(ierror)
	open (UNIT=1,FILE='valuefun_par.csv',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) Kgrid(index_k), ",", value_h(index_k), ",", value_l(index_k)
        end do
	close (UNIT=1)

        open (UNIT=1,FILE='polfun_par.csv',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) Kgrid(index_k), ",", g_k_h(index_k), ",", g_k_l(index_k)
        end do
	close (UNIT=1)

END PROGRAM
