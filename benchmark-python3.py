import subprocess, os, sys

TESTS : int = 10;

OS_NAME : str = sys.argv[1];
DATA_FOLDER : str = f"data/{OS_NAME}";
DATA_FILE : str = f"{DATA_FOLDER}/python3.csv";
if not os.path.isdir(DATA_FOLDER):
    os.mkdir(DATA_FOLDER);
input_size : int = 0;
task : str = "none";

# --------------------------- BINARY TREES ---------------------------

BINARYTREES_BUILD : str = "python3 benchmarkgame-sourcecode/binarytrees/binarytrees.python3-5.py";
task = "binarytrees";

input_size = 7;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{BINARYTREES_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

input_size = 21;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{BINARYTREES_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

# -------------------------- FANNKUCH REDUX --------------------------

FANNKUCH_BUILD : str = "python3 benchmarkgame-sourcecode/fannkuchredux/fannkuchredux.python3-8.py";
task = "fannkuchredux";

input_size = 10;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FANNKUCH_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

input_size = 12;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FANNKUCH_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

# ------------------------------- FASTA ------------------------------

FASTA_BUILD : str = "python3 benchmarkgame-sourcecode/fasta/fasta.python3-8.py";
task = "fasta";

input_size = 250000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FASTA_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

input_size = 25000000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FASTA_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

# ---------------------------- MANDELBROT ----------------------------

MANDELBROT_BUILD : str = "python3 benchmarkgame-sourcecode/mandelbrot/mandelbrot.python3-8.py";
task = "mandelbrot";

input_size = 1000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{MANDELBROT_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);

input_size = 16000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{MANDELBROT_BUILD} {input_size}", DATA_FILE, OS_NAME, "Python 3", task, str(input_size)]);
