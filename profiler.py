# ARGS: [command to test] [file to store measurements] [os name] [programming langauge] [task name] [input size]
import os, sys
import time

real_time = time.time();
status = os.system(sys.argv[1] + " > /dev/null");
real_time = time.time() - real_time;

rusage_dict = {
    "os" : sys.argv[3], # The name of the operating system
    "language" : sys.argv[4], # The name of the programming language
    "task" : sys.argv[5], # The name of the task performed
    "input_size" : sys.argv[6], # The input size for the task
    "pid" : os.getpid(), # The process id
    "status" : status, # The status the task exited with
    "rtime" : real_time, # The real time the program ran for
};

if sys.platform == "linux":
    from resource import *
    rusage = getrusage(RUSAGE_CHILDREN);
    rusage_unused_fields = {"ru_ixrss","ru_idrss","ru_isrss","ru_nswap","ru_msgsnd","ru_msgrcv","ru_nsignals"};

    for i in range(len(rusage)):
        key : str = rusage.__match_args__[i];
        if key in rusage_unused_fields:
            continue;
        rusage_dict[key] = rusage[i];
elif sys.platform == "win32":
    # WIP
    pass;


import csv

out_file = sys.argv[2];
file_exists = os.path.isfile(out_file);
with open(out_file, "a") as f:
    writer = csv.DictWriter(f, fieldnames=rusage_dict.keys());
    if not file_exists:
        writer.writeheader();
    writer.writerow(rusage_dict);

print("Complete: ", sys.argv);
print(rusage_dict);
