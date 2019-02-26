This tool generates a 21x21 covariance matrix and keeps it into an output file with extension .cov
The GenerateCovarianceMatrix.py takes in an alignment (.aln) file as an argument in command line and generates a .cov file with the same name.
For example executing the command:

	$ python GenerateCovarianceMatrix.py xyz.aln outputfilename.cov [optional]

will generate a file called "xyz.cov" if the outputfilename.cov is not given.
The file MakeCovarianceBinaryFile is an executable tool that GenerateCovarianceMatrix.py uses. This tools has been derived from the DeepCov software.

The Covariance Matrix here is a 21 x 21 matrix which represents the 20 amino acids and 1 more for a gap in the sequence. The matrix is a symmetric matrix and the 
diagonals represent the variance. 
