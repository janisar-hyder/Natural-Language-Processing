#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_WORD_LEN 100

// Function to count words in a portion of the file
int count_words_in_chunk(char *filename, int start, int end) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("File opening failed");
        return -1;
    }

    char word[MAX_WORD_LEN];
    int word_count = 0;
    fseek(file, start, SEEK_SET);

    while (ftell(file) < end && fscanf(file, "%s", word) == 1) {
        word_count++;
    }

    fclose(file);
    return word_count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: mpirun -np <num_processes> %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *filename = argv[1];
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Record the start time
    double start_time = MPI_Wtime();

    // Open the file and determine its size
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("File opening failed");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

    // Determine the portion of the file each process will handle
    long chunk_size = file_size / size;
    long start = rank * chunk_size;
    long end = (rank == size - 1) ? file_size : (rank + 1) * chunk_size;

    // Count the words in the assigned portion of the file
    int local_word_count = count_words_in_chunk(filename, start, end);

    // Gather the local counts to the root process
    int total_word_count = 0;
    MPI_Reduce(&local_word_count, &total_word_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only the root process will print the final count
    if (rank == 0) {
        printf("Total word count: %d\n", total_word_count);

        // Record the end time and calculate the duration
        double end_time = MPI_Wtime();
        printf("Time taken: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

