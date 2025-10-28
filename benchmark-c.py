import subprocess, os, sys

TESTS : int = 10;

OS_NAME : str = sys.argv[1];
DATA_FOLDER : str = f"data/{OS_NAME}";
DATA_FILE : str = f"{DATA_FOLDER}/c.csv";
if not os.path.isdir(DATA_FOLDER):
    os.mkdir(DATA_FOLDER);
input_size : int = 0;
task : str = "none";

# --------------------------- BINARY TREES ---------------------------

BINARYTREES_BUILD : str = "build/binarytrees/binarytrees-c";
os.system(f"gcc -O3 -o {BINARYTREES_BUILD} benchmarkgame-sourcecode/binarytrees/binarytrees.gcc-5.c");
task = "binarytrees";

input_size = 7;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{BINARYTREES_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

input_size = 21;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{BINARYTREES_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

# -------------------------- FANNKUCH REDUX --------------------------

FANNKUCH_BUILD : str = "build/fannkuchredux/fannkuchredux-c";
os.system(f"gcc -O3 -o {FANNKUCH_BUILD} benchmarkgame-sourcecode/fannkuchredux/fannkuchredux.gcc-8.c");
task = "fannkuchredux";

input_size = 10;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FANNKUCH_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

input_size = 12;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FANNKUCH_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

# ------------------------------- FASTA ------------------------------

FASTA_BUILD : str = "build/fasta/fasta-c";
os.system(f"gcc -O3 -o {FASTA_BUILD} benchmarkgame-sourcecode/fasta/fasta.gcc-8.c");
task = "fasta";

input_size = 250000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FASTA_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

input_size = 25000000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FASTA_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

# ---------------------------- MANDELBROT ----------------------------

MANDELBROT_BUILD : str = "build/mandelbrot/mandelbrot-c";
os.system(f"gcc -O3 -o {MANDELBROT_BUILD} benchmarkgame-sourcecode/mandelbrot/mandelbrot.gcc-9.c");
task = "mandelbrot";

input_size = 1000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{MANDELBROT_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);

input_size = 16000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{MANDELBROT_BUILD} {input_size}", DATA_FILE, OS_NAME, "C", task, str(input_size)]);
