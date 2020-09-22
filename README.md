#Matrix Addition and Multiplication
This is a cuda program that covers "Matrix Addition and Matrix Multiplication" for class.

#Compiling
nvcc was used to compile these programs. This will create a executable program.
Command for compiling matrix addition: nvcc Fleming-MatrixAdd.cu -o MatrixAdd
Command for compiling matrix multiplication: nvcc Fleming-MatrixMul.cu -o MatrixMul

#Running
These programs can be run directly from the command line. They both have the arguments MATRIXSIZE and BLOCKSIZE. A GPU and cuda is required to run the programs.
Command for matrix addition: {path}/MatrixAdd {MATRIXSIZE} {BLOCKSIZE}
Command for matrix Multiplication: {path}/MatrixMul {MATRIXSIZE} {BLOCKSIZE}
