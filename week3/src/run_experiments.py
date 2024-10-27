import subprocess

# Path to the Python interpreter in the virtual environment
venv_python = r"D:\Team4\.venv\Scripts\python.exe"

# Define the parameters for blocks, bins, similarity measures, and descriptor type
block_sizes = [64, 128, 256]
N_values = [5, 10, 100, 500, 1000]
similarity_measures = [
	"Lorentzian", "Manhattan", "HISTCMP_HELLINGER", "HISTCMP_CHISQR_ALT"
]

descriptor_type = "DCT"

data_path = ".\\data\\qsd2_w3\\masked\\images_without_noise"


# Loop over all combinations of block sizes, bin sizes, and similarity measures
for similarity in similarity_measures:
	for blocks in block_sizes:
		for N in N_values:
			# First, run the compute_db_descriptors script
			compute_command = [
				venv_python, ".\\compute_db_descriptors.py",
				"--num_blocks", str(blocks),
				"--N", str(N),
				"--descriptor_type", descriptor_type
			]
			#subprocess.run(compute_command)
			print(f"Completed compute command for blocks: {blocks}, N: {N}, similarity: {similarity}")

			# Then, run the main script to compute the results
			main_command = [
				venv_python, ".\\main.py", data_path,
				f"--num_blocks={blocks}",
				f"--similarity_measure={similarity}",
				f"--N={N}",
				f"--descriptor_type={descriptor_type}"
			]
			main_output_file = f"output_blocks_{blocks}_N_{N}_similarity_{similarity}_main.txt"

			# Execute the main command and wait for it to finish
			with open(main_output_file, "w") as main_output:
				subprocess.run(main_command, stdout=main_output, stderr=main_output)

			print(f"Saved output for blocks: {blocks}, N: {N}, similarity: {similarity} to {main_output_file}")
