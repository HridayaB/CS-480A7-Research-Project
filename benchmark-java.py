import subprocess, os, sys

TESTS : int = 10;

OS_NAME : str = sys.argv[1];
DATA_FOLDER : str = f"data/{OS_NAME}";
DATA_FILE : str = f"{DATA_FOLDER}/java.csv";
if not os.path.isdir(DATA_FOLDER):
    os.mkdir(DATA_FOLDER);
input_size : int = 0;
task : str = "none";

# --------------------------- BINARY TREES ---------------------------

BINARYTREES_BUILD : str = "cd build/binarytrees && java binarytrees";
os.system("javac -d build/binarytrees benchmarkgame-sourcecode/binarytrees/binarytrees.java");
task = "binarytrees";

input_size = 7;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{BINARYTREES_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

input_size = 21;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{BINARYTREES_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

# -------------------------- FANNKUCH REDUX --------------------------

FANNKUCH_BUILD : str = "cd build/fannkuchredux && java fannkuchredux";
os.system("javac -d build/fannkuchredux benchmarkgame-sourcecode/fannkuchredux/fannkuchredux.java");
task = "fannkuchredux";

input_size = 10;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FANNKUCH_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

input_size = 12;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FANNKUCH_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

# ------------------------------- FASTA ------------------------------

FASTA_BUILD : str = "cd build/fasta && java fasta";
os.system("javac -d build/fasta benchmarkgame-sourcecode/fasta/fasta.java");
task = "fasta";

input_size = 250000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FASTA_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

input_size = 25000000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{FASTA_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

# ---------------------------- MANDELBROT ----------------------------

MANDELBROT_BUILD : str = "cd build/mandelbrot && java mandelbrot";
os.system("javac -d build/mandelbrot benchmarkgame-sourcecode/mandelbrot/mandelbrot.java");
task = "mandelbrot";

input_size = 1000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{MANDELBROT_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);

input_size = 16000;
for i in range(TESTS):
    process = subprocess.run(["python3", "profiler.py", f"{MANDELBROT_BUILD} {input_size}", DATA_FILE, OS_NAME, "Java", task, str(input_size)]);
