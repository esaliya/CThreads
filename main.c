#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"
#include <unistd.h>

double*threadPartialBofZ;
double* preX;
double* threadPartialOutMM;
int targetDimension = 3;
int blockSize = 64;

double currentTimeInSeconds(void)
{

    int flag;
    clockid_t cid = CLOCK_REALTIME; // CLOCK_MONOTONE might be better
    struct timespec tp;
    double timing;

    flag = clock_gettime(cid, &tp);
    if (flag == 0) timing = tp.tv_sec + 1.0e-9*tp.tv_nsec;
    else           timing = -17.0;         // If timer failed, return non-valid time

    return(timing);
};
void matrixMultiply(double* A, double* B, int aHeight, int bWidth, int comm, int bz, double* C, int threadAOffset, int threadCOffset) {

    int aHeightBlocks = aHeight / bz; // size = Height of A
    int aLastBlockHeight = aHeight - (aHeightBlocks * bz);
    if (aLastBlockHeight > 0) {
        aHeightBlocks++;
    }

    int bWidthBlocks = bWidth / bz; // size = Width of B
    int bLastBlockWidth = bWidth - (bWidthBlocks * bz);
    if (bLastBlockWidth > 0) {
        bWidthBlocks++;
    }

    int commnBlocks = comm / bz; // size = Width of A or Height of B
    int commLastBlockWidth = comm - (commnBlocks * bz);
    if (commLastBlockWidth > 0) {
        commnBlocks++;
    }

    int aBlockHeight = bz;
    int bBlockWidth;
    int commBlockWidth;

    int ib, jb, kb, i, j, k;
    int iARowOffset, kBRowOffset, iCRowOffset;
    for (ib = 0; ib < aHeightBlocks; ib++) {
        if (aLastBlockHeight > 0 && ib == (aHeightBlocks - 1)) {
            aBlockHeight = aLastBlockHeight;
        }
        bBlockWidth = bz;
        for (jb = 0; jb < bWidthBlocks; jb++) {
            if (bLastBlockWidth > 0 && jb == (bWidthBlocks - 1)) {
                bBlockWidth = bLastBlockWidth;
            }
            commBlockWidth = bz;
            for (kb = 0; kb < commnBlocks; kb++) {
                if (commLastBlockWidth > 0 && kb == (commnBlocks - 1)) {
                    commBlockWidth = commLastBlockWidth;
                }

                for (i = ib * bz; i < (ib * bz) + aBlockHeight; i++) {
                    iARowOffset = i*comm+threadAOffset;
                    iCRowOffset = i*bWidth+threadCOffset;
                    for (j = jb * bz; j < (jb * bz) + bBlockWidth;
                         j++) {
                        for (k = kb * bz;
                             k < (kb * bz) + commBlockWidth; k++) {
                            kBRowOffset = k*bWidth;
                            if (A[iARowOffset+k] != 0 && B[kBRowOffset+j] != 0) {
                                C[iCRowOffset+j] += A[iARowOffset+k] * B[kBRowOffset+j];
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
void matrixMultiply(double** A, double** B, int aHeight, int bWidth, int comm, int bz, double** C) {

    int aHeightBlocks = aHeight / bz; // size = Height of A
    int aLastBlockHeight = aHeight - (aHeightBlocks * bz);
    if (aLastBlockHeight > 0) {
        aHeightBlocks++;
    }

    int bWidthBlocks = bWidth / bz; // size = Width of B
    int bLastBlockWidth = bWidth - (bWidthBlocks * bz);
    if (bLastBlockWidth > 0) {
        bWidthBlocks++;
    }

    int commnBlocks = comm / bz; // size = Width of A or Height of B
    int commLastBlockWidth = comm - (commnBlocks * bz);
    if (commLastBlockWidth > 0) {
        commnBlocks++;
    }

    int aBlockHeight = bz;
    int bBlockWidth = bz;
    int commBlockWidth = bz;

    int ib, jb, kb, i, j, k;
    for (ib = 0; ib < aHeightBlocks; ib++) {
        if (aLastBlockHeight > 0 && ib == (aHeightBlocks - 1)) {
            aBlockHeight = aLastBlockHeight;
        }
        bBlockWidth = bz;
        commBlockWidth = bz;
        for (jb = 0; jb < bWidthBlocks; jb++) {
            if (bLastBlockWidth > 0 && jb == (bWidthBlocks - 1)) {
                bBlockWidth = bLastBlockWidth;
            }
            commBlockWidth = bz;
            for (kb = 0; kb < commnBlocks; kb++) {
                if (commLastBlockWidth > 0 && kb == (commnBlocks - 1)) {
                    commBlockWidth = commLastBlockWidth;
                }

                for (i = ib * bz; i < (ib * bz) + aBlockHeight; i++) {
                    for (j = jb * bz; j < (jb * bz) + bBlockWidth;
                         j++) {
                        for (k = kb * bz;
                             k < (kb * bz) + commBlockWidth; k++) {
                            if (A[i][k] != 0 && B[k][j] != 0) {
                                C[i][j] += A[i][k] * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    }
}
*/

void MMMpi(int threadCount, int iterations, int globalColCount, int nodesPerNode) {

    int worldProcRank, worldProcCount;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldProcCount);
    int rowCountPerUnit = globalColCount / worldProcCount;
    int pointComponentCountGlobal = globalColCount * targetDimension;
    int pointComponentCountLocal = rowCountPerUnit * targetDimension;
    preX = (double*) malloc(sizeof(double) * pointComponentCountGlobal);
    int i;

    int pairCountLocal = rowCountPerUnit * globalColCount;
    threadPartialBofZ = (double*) malloc(sizeof(double) * threadCount * pairCountLocal);
    threadPartialOutMM = (double*) malloc(sizeof(double*)*threadCount*pointComponentCountLocal);
    int j;


    double time = 0.0;
    double compTime = 0.0;
    double commTime = 0.0;

    int itr;
    int k;
    for (itr = 0; itr < iterations; ++itr){
        for (i = 0; i < pointComponentCountGlobal; ++i){
            preX[i] = (double)rand() / (double)RAND_MAX;
        }

        int pairCountAllThreads = threadCount*pairCountLocal;
        for (k = 0; k < pairCountAllThreads; ++k){
            threadPartialBofZ[k] = (double)rand() / (double)RAND_MAX;
        }

        int pointComponentCountAllThreads = threadCount*pointComponentCountLocal;
        for (k = 0; k < pointComponentCountAllThreads; ++k){
            threadPartialOutMM[k] = (double)0.0;
        }


        if (threadCount == 1) {
            double t1;
            const int threadIdx = 0;
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = currentTimeInSeconds();
            matrixMultiply(threadPartialBofZ, preX, rowCountPerUnit, targetDimension, globalColCount, blockSize,
                           threadPartialOutMM, threadIdx * pairCountLocal, threadIdx * pointComponentCountLocal);
            time = currentTimeInSeconds() - t1;

            MPI_Barrier(MPI_COMM_WORLD);
            t1 = currentTimeInSeconds();

            if (worldProcRank == 0) {
                printf("RowCount %d ColCount %d, itr %d time %lf compute %lf comm %lf\n", rowCountPerUnit,
                       globalColCount, itr, time * 1000, compTime * 1000, commTime * 1000);
            }
        }
    }


}


int main(int argc, char **args) {
    MPI_Init(&argc, &args);
    if (argc < 4) {
        printf("We need 3 arguments");
        exit(1);
    }

    double t1 = currentTimeInSeconds();
    sleep(4);
    double t2 = currentTimeInSeconds();
    printf("%lf s\n",(t2-t1));

    /* Take these as command line args
     * 1. iterations -- i
     * 2. colcount -- c
     * 3. nodes per node  -- n*/
    int t = 1;
    int i = 0;
    int c = 0;
    int n = 0;

    i = atoi(args[1]);
    c = atoi(args[2]);
    n = atoi(args[3]);

    MMMpi(t, i, c, n);
    MPI_Finalize();

    return 0;
}